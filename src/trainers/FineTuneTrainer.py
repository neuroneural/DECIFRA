"""
Single-stage fine-tuning trainer: one train/val/test cycle for a (fold, repeat).

Reuses the model's ``handle_batch`` (classification path), keeps the best-by-val
weights in memory via EarlyStopping, and writes resumable checkpoints. The outer
nested-CV loop lives in scripts/02_finetuning.py; this trainer runs one cell.
"""
import os
import time

import torch
import pandas as pd
from omegaconf import OmegaConf


class EarlyStopping:
    """Tracks the best-by-val checkpoint in memory; stops after `patience` misses."""

    def __init__(self, patience: int, minimize: bool = True):
        self.patience = patience
        self.minimize = minimize
        self.best = None
        self.best_epoch = -1
        self.best_state = None
        self.counter = 0
        self.stop = False

    def step(self, score, model, epoch):
        improved = self.best is None or (
            score < self.best if self.minimize else score > self.best
        )
        if improved:
            self.best = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved


class FineTuneTrainer:
    def __init__(
        self,
        cfg,
        model_cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        epochs: int = 200,
        patience: int = None,
        device: str = None,
        save_path: str = None,
        resume: bool = False,
        preserve_checkpoints: bool = False,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        # `patience: null` in config => no early stop (train full, keep best-by-val)
        self.patience = patience if patience is not None else epochs
        self.resume = resume
        self.preserve_checkpoints = preserve_checkpoints

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("mps") if torch.backends.mps.is_available() \
                else torch.device("cpu")
        self.model.to(self.device)
        self.SAVE_PATH = save_path

    def _epoch(self, loader, train: bool):
        # NB: unlike the pretrainers we never call model.set_epoch here — a staged
        # model is pinned once via setup_finetuning() and must not re-ramp.
        with torch.set_grad_enabled(train):
            self.model.train(train)
            total_loss = 0.0
            n_batches = len(loader)
            agg = {}
            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                loss, log = self.model.handle_batch(batch)
                total_loss += float(loss.detach().cpu().item())
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                for k, v in log.items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0.0) + float(v)
            agg = {k: v / n_batches for k, v in agg.items()}
            agg["loss"] = total_loss / n_batches
            return agg

    def is_complete(self):
        """A finished run has best.pt written (also serves as the run marker)."""
        return os.path.exists(os.path.join(self.SAVE_PATH, "best.pt"))

    def run(self):
        ckpt_dir = os.path.join(self.SAVE_PATH, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(self.SAVE_PATH, "model_config.yaml"), "w") as f:
            OmegaConf.save(config=self.model_cfg, f=f.name)
        with open(os.path.join(self.SAVE_PATH, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.cfg, f=f.name)

        # Staged models pin their cross-channel scale once for single-stage FT.
        if hasattr(self.model, "setup_finetuning"):
            self.model.setup_finetuning()

        log_path = os.path.join(self.SAVE_PATH, "train_logs.csv")
        stopper = EarlyStopping(self.patience, minimize=True)
        train_logs = []
        start_epoch = 0
        write_header = True

        if self.resume and os.path.exists(log_path):
            prev = pd.read_csv(log_path)
            if not prev.empty:
                train_logs = prev.to_dict("records")
                start_epoch = int(prev.iloc[-1]["epoch"]) + 1
                write_header = False
                last = os.path.join(ckpt_dir, "last.pt")
                if os.path.exists(last):
                    self.model.load_state_dict(torch.load(last))
                opt = os.path.join(ckpt_dir, "optimizer_last.pt")
                if os.path.exists(opt):
                    self.optimizer.load_state_dict(torch.load(opt))
                # re-seed best-so-far from logged val_loss (best weights recovered
                # from a per-epoch ckpt if preserved, else from the last state)
                bi = int(prev["val_loss"].idxmin())
                stopper.best = float(prev["val_loss"].iloc[bi])
                stopper.best_epoch = int(prev["epoch"].iloc[bi])
                be_ckpt = os.path.join(ckpt_dir, f"model_{stopper.best_epoch}.pt")
                src = be_ckpt if os.path.exists(be_ckpt) else last
                if os.path.exists(src):
                    stopper.best_state = torch.load(src)
                print(f"Resuming fine-tune from epoch {start_epoch}")

        start = time.time()
        for epoch in range(start_epoch, self.epochs):
            tr = self._epoch(self.train_loader, train=True)
            va = self._epoch(self.val_loader, train=False)

            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer_last.pt"))
            if self.preserve_checkpoints:
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}.pt"))

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in tr.items()})
            row.update({f"val_{k}": v for k, v in va.items()})
            train_logs.append(row)
            pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=write_header)
            write_header = False

            stopper.step(va["loss"], self.model, epoch)
            if stopper.stop:
                print(f"Early stop at epoch {epoch} (best @ {stopper.best_epoch})")
                break

        # Restore best-by-val weights, persist as best.pt (run-complete marker).
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        torch.save(self.model.state_dict(), os.path.join(self.SAVE_PATH, "best.pt"))
        with open(os.path.join(self.SAVE_PATH, "best_epoch.txt"), "w") as f:
            f.write(f"{stopper.best_epoch}\n")

        # Test with the best model.
        test = self._epoch(self.test_loader, train=False)
        test_row = {f"test_{k}": v for k, v in test.items()}
        test_row["best_epoch"] = stopper.best_epoch
        test_row["train_time_s"] = round(time.time() - start, 1)
        pd.DataFrame([test_row]).to_csv(
            os.path.join(self.SAVE_PATH, "test_log.csv"), index=False
        )

        if not self.preserve_checkpoints:
            for fn in ("last.pt", "optimizer_last.pt"):
                p = os.path.join(ckpt_dir, fn)
                if os.path.exists(p):
                    os.remove(p)

        return test_row
