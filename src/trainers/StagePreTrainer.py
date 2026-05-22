import torch
import os
import time
import pandas as pd
from omegaconf import OmegaConf
from src.trainers.BasicPreTrainer import BasicPreTrainer

class StagePreTrainer(BasicPreTrainer):
    """
    Modular trainer that runs through multiple training stages sequentially.
    Each stage has its own training loop and best-epoch finding.
    """
    def __init__(
        self,
        cfg,
        model_cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
        stage_epochs_list: list,
        scheduler=None,
        device: str = None,
        save_path: str = None,
        resume: bool = False,
    ) -> None:
        # Total epochs is the sum of all stages
        self.stage_epochs_list = stage_epochs_list
        super().__init__(
            cfg, model_cfg, model, optimizer, train_loader, val_loader, 
            scheduler=scheduler, epochs=sum(stage_epochs_list), device=device, save_path=save_path, resume=resume
        )

    def run(self):
        # Initial backup logic
        if not self.resume and os.path.exists(self.SAVE_PATH) and os.listdir(self.SAVE_PATH):
            utc_string = time.strftime("%m%d-%H%M%S", time.gmtime())
            backup_path = f"{self.SAVE_PATH}_{utc_string}"
            print(f"Save path {self.SAVE_PATH} already exists. Backing up to {backup_path}")
            os.makedirs(backup_path, exist_ok=True)
            for filename in os.listdir(self.SAVE_PATH):
                os.rename(os.path.join(self.SAVE_PATH, filename), os.path.join(backup_path, filename))
                
        checkpoints_path = os.path.join(self.SAVE_PATH, "checkpoints")
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)
        
        with open(os.path.join(self.SAVE_PATH, "model_config.yaml"), "w") as f:
            OmegaConf.save(config=self.model_cfg, f=f.name)
        with open(os.path.join(self.SAVE_PATH, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.cfg, f=f.name)

        all_stage_logs = []
        start_time = time.time()
        log_path = os.path.join(self.SAVE_PATH, "train_logs.csv")
        write_header = not os.path.exists(log_path) or os.stat(log_path).st_size == 0
        
        start_cumulative_epoch = 0
        
        if self.resume and os.path.exists(log_path):
            existing_logs = pd.read_csv(log_path)
            if not existing_logs.empty:
                all_stage_logs = existing_logs.to_dict('records')
                last_epoch = int(existing_logs.iloc[-1]['epoch'])
                start_cumulative_epoch = last_epoch + 1
                
                print(f"Resuming from total epoch {start_cumulative_epoch} (found logs up to epoch {last_epoch})")
                
                # Load model
                model_ckpt = os.path.join(checkpoints_path, f"model_{last_epoch}.pt")
                if os.path.exists(model_ckpt):
                    self.model.load_state_dict(torch.load(model_ckpt))
                    print(f"Loaded model state from {model_ckpt}")
                
                # Load optimizer
                opt_ckpt = os.path.join(checkpoints_path, "optimizer_last.pt")
                if os.path.exists(opt_ckpt):
                    self.optimizer.load_state_dict(torch.load(opt_ckpt))
                    print(f"Loaded optimizer state from {opt_ckpt}")
                else:
                    print(f"No optimizer state found at {opt_ckpt}. Proceeding with fresh optimizer (loss may slightly spike).")

        if start_cumulative_epoch == 0:
            torch.save(self.model.state_dict(), os.path.join(checkpoints_path, "model_init.pt"))
        
        cumulative_epoch = 0
        for stage_idx, num_epochs in enumerate(self.stage_epochs_list):
            # Fast-forward stages if we already completed them
            if cumulative_epoch + num_epochs <= start_cumulative_epoch:
                cumulative_epoch += num_epochs
                continue
                
            start_stage_epoch = 0
            if cumulative_epoch < start_cumulative_epoch:
                start_stage_epoch = start_cumulative_epoch - cumulative_epoch
                cumulative_epoch = start_cumulative_epoch

            print(f"\n--- Starting Stage {stage_idx+1}/{len(self.stage_epochs_list)} ({num_epochs} epochs, starting from {start_stage_epoch}) ---")
            
            # Notify model of stage change
            if hasattr(self.model, "set_stage"):
                self.model.set_stage(stage_idx, num_epochs)
            
            # Repopulate stage_logs for finding best_epoch later
            stage_logs = [log for log in all_stage_logs if log['stage'] == stage_idx]
            
            for stage_epoch in range(start_stage_epoch, num_epochs):
                print(f"Stage {stage_idx+1} | Epoch {stage_epoch+1}/{num_epochs} (Total: {cumulative_epoch+1}) | Elapsed: {time.time()-start_time:.0f}s")
                
                # BasicPreTrainer's _epoch handles set_epoch internally now
                train_log = self._epoch(self.train_loader, train=True, epoch_idx=stage_epoch)
                val_log = self._epoch(self.val_loader, train=False, epoch_idx=stage_epoch)

                torch.save(self.model.state_dict(), os.path.join(checkpoints_path, f"model_stage{stage_idx}_epoch{stage_epoch}.pt"))
                # Also save with total epoch idx for easier tracking across stages
                torch.save(self.model.state_dict(), os.path.join(checkpoints_path, f"model_{cumulative_epoch}.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join(checkpoints_path, "optimizer_last.pt"))

                epoch_log = {
                    "model": self.model.__class__.__name__, 
                    "epoch": cumulative_epoch,
                    "stage": stage_idx,
                    "stage_epoch": stage_epoch
                }
                epoch_log.update({f"train_{k}": v for k, v in train_log.items()})
                epoch_log.update({f"val_{k}": v for k, v in val_log.items()})
                stage_logs.append(epoch_log)
                all_stage_logs.append(epoch_log)

                # Save to CSV (incremental)
                df = pd.DataFrame([epoch_log])
                df.to_csv(log_path, mode='a', index=False, header=write_header)
                write_header = False
                
                cumulative_epoch += 1

            # Find best epoch for this stage
            stage_df = pd.DataFrame(stage_logs)
            best_idx = stage_df['val_loss'].idxmin()
            best_stage_epoch = stage_df.loc[best_idx, 'stage_epoch']
            best_total_epoch = stage_df.loc[best_idx, 'epoch']
            
            with open(os.path.join(self.SAVE_PATH, f"best_epoch_stage{stage_idx}.txt"), "w") as f:
                f.write(f"stage_epoch: {best_stage_epoch}\ntotal_epoch: {best_total_epoch}\n")
            
            # load this best checkpoint
            self.model.load_state_dict(torch.load(os.path.join(checkpoints_path, f"model_stage{stage_idx}_epoch{best_stage_epoch}.pt")))
            
            print(f"Stage {stage_idx+1} complete. Best val_loss at total epoch {best_total_epoch}")

        # Final best epoch overall
        all_df = pd.DataFrame(all_stage_logs)
        best_overall_idx = all_df['val_loss'].idxmin()
        best_overall_total_epoch = all_df.loc[best_overall_idx, 'epoch']
        with open(os.path.join(self.SAVE_PATH, "best_epoch.txt"), "w") as f:
            f.write(f"{best_overall_total_epoch}\n")

        print(f"\nMultistage training complete. Overall best epoch: {best_overall_total_epoch}")
        return all_df
