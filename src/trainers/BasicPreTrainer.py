# trainer class
from omegaconf import OmegaConf
import torch
import pandas as pd

import time
import os

from src.settings import LOGS_ROOT

class BasicPreTrainer:

    def __init__(
        self,
        cfg,
        model_cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler = None,
        epochs: int = 100,
        device: str = None,
        save_path: str = None,
    ) -> None:
        
        self.cfg = cfg
        self.model_cfg = model_cfg

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("mps") if torch.backends.mps.is_available() \
                    else torch.device("cpu")
            
        # move model to device
        self.model.to(self.device)

        # set up save path
        if save_path is None:
            utc_string = time.strftime("%m%d-%H%M%S", time.gmtime())
            save_path = f"1_pretrain_{self.model.__class__.__name__}_{utc_string}"
            save_path = os.path.join(LOGS_ROOT, save_path)
        self.save_path = save_path


    def _epoch(self, loader, train: bool):
        with torch.set_grad_enabled(train):
            self.model.train(train)
            total_loss = 0.0
            n_batches = len(loader)
            agg_log = {}

            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                loss, batch_log = self.model.handle_batch(batch)

                total_loss += float(loss.detach().cpu().item())

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                for k, v in batch_log.items():
                    if isinstance(v, (int, float)):
                        agg_log[k] = agg_log.get(k, 0.0) + float(v)

            # average logs over batches
            agg_log = {k: v / n_batches for k, v in agg_log.items()}
            agg_log["loss"] = total_loss / n_batches
            return agg_log

    def run(self):
        # set save path, save config
        checkpoints_path = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)
        with open(os.path.join(self.save_path, "model_config.yaml"), "w") as f:
            OmegaConf.save(config=self.model_cfg, f=f.name)
        with open(os.path.join(self.save_path, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.cfg, f=f.name)

        ### Training loop
        train_logs = []
        start = time.time()
        torch.save(self.model.state_dict(), os.path.join(checkpoints_path, "model_init.pt"))
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs} | Elapsed time: {time.time()-start:.0f}s")
            train_log = self._epoch(self.train_loader, train=True)
            val_log = self._epoch(self.val_loader, train=False)

            torch.save(self.model.state_dict(), os.path.join(checkpoints_path, f"model_{epoch}.pt"))

            # handle logs
            epoch_log = {"model": self.model.__class__.__name__, "epoch": epoch}
            epoch_log.update({f"train_{k}": v for k, v in train_log.items()})
            epoch_log.update({f"val_{k}": v for k, v in val_log.items()})
            train_logs.append(epoch_log)

            print(epoch_log)

        # find best epoch
        train_logs = pd.DataFrame(train_logs)
        best_epoch = train_logs['val_loss'].idxmin()
        with open(os.path.join(self.save_path, "best_epoch.txt"), "w") as f:
            f.write(f"{best_epoch}\n")
        
        # build train history DataFrame
        train_logs = pd.DataFrame(train_logs)
        train_logs.to_csv(os.path.join(self.save_path, "train_logs.csv"), index=False)

        return train_logs
    