# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" GRU Forecaster baseline (Standard Multivariante GRU) """

import torch
from torch import nn
from torch.nn.functional import mse_loss
from omegaconf import OmegaConf, DictConfig

from src.models.BaseModel import BaseModel

def default_HPs(cfg: DictConfig):
    model_cfg = {
        "rnn": {
            "hidden_size": 210,
            "num_layers": 2,
            "dropout": 0.5,
        },
        "loss": {
            "prediction_delay": 0,
            "prediction_depth": 1,
            "forecast_weight": 1.0, 
        },
        "lr": 5e-4,
        "input_size": cfg.data_info.feature_size,
    }
    return OmegaConf.create(model_cfg)


def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.input_size = cfg.data_info.feature_size
    return model_cfg


class GRU_forecaster(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(GRU_forecaster, self).__init__()

        self.model_cfg = model_cfg
        self.lr = model_cfg.lr
        
        C = model_cfg.input_size
        H = model_cfg.rnn.hidden_size
        num_layers = model_cfg.rnn.num_layers
        dropout = model_cfg.rnn.dropout
        
        self.gru = nn.GRU(
            input_size=C, 
            hidden_size=H, 
            batch_first=True, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.predictor = nn.Linear(H, C)

    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def forward(self, x): 
        B, T, C = x.shape
        x = x.contiguous()
        
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        
        # Base prediction from t_0 to T_minus_1
        hidden_states = gru_out[:, :-1, :].contiguous()
        pred_curr = self.predictor(hidden_states)

        depth = self.model_cfg.loss.prediction_depth
        all_preds = [pred_curr]
        
        num_layers, H = self.model_cfg.rnn.num_layers, self.model_cfg.rnn.hidden_size
        T_minus_1 = hidden_states.size(1)
        
        # Recursive forecasting for depth > 1
        h_curr = gru_out[:, :-1, :].reshape(B * T_minus_1, H).unsqueeze(0).repeat(num_layers, 1, 1).contiguous()

        for d in range(1, depth):
            # Input for next step is the previous prediction
            x_step = pred_curr.reshape(B * T_minus_1, 1, C)
            
            self.gru.flatten_parameters()
            gru_out_step, h_curr = self.gru(x_step, h_curr)
            
            pred_curr_flat = self.predictor(gru_out_step.squeeze(1))
            pred_curr = pred_curr_flat.reshape(B, T_minus_1, C)
            
            # Truncate states exceeding timeline
            padded_pred = torch.zeros(B, T_minus_1, C, device=pred_curr.device)
            padded_pred[:, :T_minus_1 - d, :] = pred_curr[:, :-d, :]
            
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
