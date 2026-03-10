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
        self.SAVE_PATH = save_path


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
        # if save path already exists and not empty, back up the contents in a timestamped folder
        if os.path.exists(self.SAVE_PATH) and os.listdir(self.SAVE_PATH):
            utc_string = time.strftime("%m%d-%H%M%S", time.gmtime())
            backup_path = f"{self.SAVE_PATH}_{utc_string}"
            print(f"Save path {self.SAVE_PATH} already exists and is not empty. Backing up contents to {backup_path}")
            os.makedirs(backup_path, exist_ok=True)
            for filename in os.listdir(self.SAVE_PATH):
                os.rename(
                    os.path.join(self.SAVE_PATH, filename),
                    os.path.join(backup_path, filename)
                )
                
        # set save path, save config
        checkpoints_path = os.path.join(self.SAVE_PATH, "checkpoints")
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)
        with open(os.path.join(self.SAVE_PATH, "model_config.yaml"), "w") as f:
            OmegaConf.save(config=self.model_cfg, f=f.name)
        with open(os.path.join(self.SAVE_PATH, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.cfg, f=f.name)

        ### Training loop
        train_logs = []
        start = time.time()
        log_path = None # used as a flag too
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

            # save epoch log to csv
            df = pd.DataFrame([epoch_log])
            if log_path is None:
                log_path = os.path.join(self.SAVE_PATH, "train_logs.csv")
                df.to_csv(log_path, mode='a', index=False, header=True)
            else:
                df.to_csv(log_path, mode='a', index=False, header=False)


        # find best epoch
        train_logs = pd.DataFrame(train_logs)
        best_epoch = train_logs['val_loss'].idxmin()
        with open(os.path.join(self.SAVE_PATH, "best_epoch.txt"), "w") as f:
            f.write(f"{best_epoch}\n")

        return train_logs
    
def find_optimal_batch_size(model, train_data, device, starting_batch_size=2, max_batch_size=8192):
    """
    Finds the maximum batch size that fits in memory for a given model and dataset.
    """
    import gc
    import torch
    
    print(f"Starting smart batch size detection on {device}...")
    model.to(device)
    optimizer = model.get_optimizer()
    
    current_batch_size = starting_batch_size
    optimal_batch_size = starting_batch_size
    
    while current_batch_size <= max_batch_size:
        try:
            tested_batch_size = min(current_batch_size, len(train_data))
            dummy_data = train_data[:tested_batch_size]
            dummy_loader = model.prepare_pretraining_dataloader(dummy_data, shuffle=False, batch_size=tested_batch_size, zscore=False)
            batch = next(iter(dummy_loader))
            batch = [b.to(device) for b in batch] if isinstance(batch, (list, tuple)) else batch.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            loss, _ = model.handle_batch(batch)
            loss.backward()
            optimizer.step()
            
            optimal_batch_size = tested_batch_size
            
            del dummy_loader, batch, loss, dummy_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            if optimal_batch_size == len(train_data):
                break
                
            current_batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower() or "memory" in str(e).lower() or "allocate" in str(e).lower():
                print(f"Memory limit reached at batch size {current_batch_size}.")
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                break
            else:
                raise e

    # Scale down by a small safety factor to avoid OOM during actual training with slightly different sequence lengths etc.
    final_batch_size = max(1, int(optimal_batch_size * 0.8))
    final_batch_size = min(final_batch_size, len(train_data))
    
    optimizer.zero_grad()
    
    return final_batch_size