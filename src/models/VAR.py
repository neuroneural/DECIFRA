# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" Linear Vector Autoregression (VAR) model for forecasting """

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import mse_loss

from omegaconf import OmegaConf, DictConfig
from src.models.BaseModel import BaseModel

def default_HPs(cfg: DictConfig):
    model_cfg = {
        "lag": 5, 
        "linear_mode": "full",  # Options: "full" (lag*C -> C), "single" (shared lag->1), "multi" (independent lag->1 per channel)
        "loss": {
            "prediction_delay": 0,
            "prediction_depth": 1,
            "forecast_weight": 1.0, 
        },
        "lr": 1e-3,
        "input_size": cfg.data_info.feature_size,
    }
    return OmegaConf.create(model_cfg)


def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    """Load custom hyperparams and inject dynamic feature_size."""
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.input_size = cfg.data_info.feature_size
    return model_cfg


class VAR(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(VAR, self).__init__()

        self.model_cfg = model_cfg
        self.lr = model_cfg.lr
        self.lag = model_cfg.lag
        self.C = model_cfg.input_size
        self.mode = model_cfg.get("linear_mode", "full")
        
        # 1. Linear Predictor
        if self.mode == "full":
            self.linear = nn.Linear(self.lag * self.C, self.C)
        elif self.mode == "single":
            # Shared AR
            self.linear = nn.Linear(self.lag, 1)
        elif self.mode == "multi":
            # Independent AR parameters per channel
            self.linear = nn.ModuleList([nn.Linear(self.lag, 1) for _ in range(self.C)])
        else:
            raise ValueError(f"Unknown linear_mode: {self.mode}. Choose from 'full', 'single', 'multi'.")


    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        # We only do forecasting with this model
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def predict_window(self, window):
        """Passes a (B, Lag, C) historical window through the linear predictor."""
        B = window.size(0)
        window = window.contiguous()
        
        if self.mode == "full":
            # Flatten lag and channels: (B, Lag * C)
            flat_window = window.reshape(B, -1).contiguous()
            return self.linear(flat_window)
        elif self.mode == "single":
            # Shared AR: map (B, Lag, C) -> (B*C, Lag) -> Linear(Lag, 1) -> (B, C)
            flat_window = window.transpose(1, 2).reshape(B * self.C, self.lag).contiguous()
            preds = self.linear(flat_window)
            return preds.reshape(B, self.C).contiguous()
        else:
            # Independent AR (multi) predictions
            preds = []
            for i in range(self.C):
                channel_window = window[:, :, i].contiguous()
                preds.append(self.linear[i](channel_window))
            return torch.cat(preds, dim=1).contiguous()


    def forward(self, x): 
        B, T, C = x.shape
        x = x.contiguous()
        
        # We need to predict t+1 using [t-lag+1 : t]
        # For the first (lag-1) steps, we lack historical data. 
        # We zero-pad the beginning of the sequence by (lag-1) along the Time dimension to standardize outputs to (T-1) perfectly.
        if self.lag > 1:
            # pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            # We are padding the Time dimension (dim=1) exclusively by lag-1 at the front.
            padded_x = F.pad(x.transpose(1, 2), (self.lag - 1, 0)).transpose(1, 2).contiguous()
        else:
            padded_x = x

        # 1) Base predictions for all standard sequence steps (T-1 forecasts)
        base_preds = []
        for t in range(self.lag - 1, (T - 1) + self.lag - 1):
            window = padded_x[:, t - self.lag + 1 : t + 1, :]
            pred_t = self.predict_window(window)
            base_preds.append(pred_t)
            
        # Shape: (B, T-1, C)
        pred_curr = torch.stack(base_preds, dim=1).contiguous()
        all_preds = [pred_curr]

        # 2) Recursive Prediction Depth Loop
        depth = self.model_cfg.loss.prediction_depth
        T_minus_1 = T - 1

        for d in range(1, depth):
            step_preds_list = []
            
            # For each sliding window step along the predicted baseline
            for t in range(T_minus_1):
                # We build a localized window consisting of:
                # 1. Any remaining real data from before 't'
                # 2. Accumulated recursive forecasts from previous depth loops up to 'd'
                
                # Fetch original padded history ending at 't'
                historical_t = t + self.lag - 1
                real_window = padded_x[:, historical_t - self.lag + 1 : historical_t + 1, :]
                
                # Overlay forecasts incrementally 
                combo_window = real_window.clone().detach() # Safe detach, building temporary matrix
                for backtrack in range(min(d, self.lag)):
                    # Replace real data with the recursive forecast we generated inside this loop
                    combo_window[:, -(backtrack + 1), :] = all_preds[d - 1 - backtrack][:, t, :]
                    
                step_pred = self.predict_window(combo_window)
                step_preds_list.append(step_pred)
                
            pred_curr_step = torch.stack(step_preds_list, dim=1).contiguous()
            
            # Truncate states exceeding timeline. E.g. t+2 cannot be forecasted at T-1
            padded_pred = torch.zeros(B, T_minus_1, C, device=x.device).contiguous()
            padded_pred[:, :T_minus_1 - d, :] = pred_curr_step[:, :-d, :]
            
            all_preds.append(padded_pred)

        return None, {"predicted": torch.stack(all_preds, dim=-1).contiguous(), "originals": x.contiguous()}


    def compute_loss(self, loss_load, targets):
        delay = self.model_cfg.loss.prediction_delay
        depth = self.model_cfg.loss.prediction_depth
        forecast_weight = self.model_cfg.loss.forecast_weight

        originals, predicted = loss_load["originals"], loss_load["predicted"]
        T, total_loss = originals.size(1), 0.0
        
        for d in range(depth):
            total_shift = delay + d
            if 1 + total_shift >= T: break
                
            target_signal = originals[:, 1 + total_shift:, :].contiguous()
            end_idx = (T - 1) - total_shift
            pred_d = predicted[:, delay:delay+end_idx, :, d].contiguous()
            
            total_loss += mse_loss(pred_d, target_signal)

        total_loss = forecast_weight * (total_loss / depth)
        return total_loss, {"forecast_loss": total_loss.item()}

    def handle_batch(self, batch):
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        _, loss_load = self.forward(data)
        loss, batch_log = self.compute_loss(loss_load, None)
        return loss, batch_log
