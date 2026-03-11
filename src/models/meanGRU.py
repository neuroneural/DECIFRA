# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" meanGRU model for forecasting """

import torch
from torch import nn
from torch.nn.functional import mse_loss

from omegaconf import OmegaConf, DictConfig

from src.models.BaseModel import BaseModel, compute_metrics


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "single_embedder": True, 
        "single_GRU": True, 
        "rnn": {
            "input_embedding_size": 16,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "single_predictor": True, 
        
        "loss": {
            "prediction_delay": 0,
            "prediction_depth": 1,
            "forecast_weight": 1.0, 
        },
        "lr": 1e-3,
        "load_pretrained": False,
        "pretrained_path": None,
        "input_size": cfg.data_info.feature_size,
    }
    return OmegaConf.create(model_cfg)


def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    """Load custom hyperparams and inject dynamic feature_size."""
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.input_size = cfg.data_info.feature_size
    return model_cfg


class meanGRU(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(meanGRU, self).__init__()

        self.model_cfg = model_cfg
        self.lr = model_cfg.lr
        
        C = model_cfg.input_size
        E = model_cfg.rnn.input_embedding_size
        H = model_cfg.rnn.hidden_size
        
        num_layers = model_cfg.rnn.num_layers
        dropout = model_cfg.rnn.dropout
        
        # 1. Embedder
        if model_cfg.single_embedder:
            self.embedder = nn.Linear(1, E)
        else:
            self.embedder = nn.ModuleList([nn.Linear(1, E) for _ in range(C)])

        # 2. GRU
        if model_cfg.single_GRU:
            self.gru = nn.GRU(
                E, H, batch_first=True, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            self.gru = nn.ModuleList([
                nn.GRU(
                    E, H, batch_first=True, num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0
                ) for _ in range(C)
            ])

        # 3. Predictor (forecasting)
        if model_cfg.single_predictor:
            self.predictor = nn.Linear(H, 1)
        else:
            self.predictor = nn.ModuleList([nn.Linear(H, 1) for _ in range(C)])


    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        # We only do forecasting with this model, so we can ignore labels or treat as TS_only if we want.
        # Following DECIFRA standard, we use base "TS" which gives (data, labels) even if labels are ignored.
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def embed_signals(self, x):
        """Pass inputs through the embedder(s). Shape: [B, T, C] or [B, C]"""
        B, C = x.size(0), self.model_cfg.input_size
        if x.dim() == 2: x = x.unsqueeze(1)
        x = x.contiguous()
            
        if self.model_cfg.single_embedder:
            return self.embedder(x.unsqueeze(-1).contiguous())
        
        return torch.stack([self.embedder[i](x[:, :, i].contiguous().unsqueeze(-1)) for i in range(C)], dim=2).contiguous()

    def run_gru(self, x_emb, h_in=None):
        """Run GRU. Returns gru_out: [B, T, C, H], h_out: [L, B, C, H]"""
        B, T, C, E = x_emb.shape
        H, num_layers = self.model_cfg.rnn.hidden_size, self.model_cfg.rnn.num_layers

        x_emb = x_emb.contiguous()
        if h_in is not None: h_in = h_in.contiguous()

        if self.model_cfg.single_GRU:
            x_input = x_emb.transpose(1, 2).reshape(B * C, T, E)
            h_input = h_in.reshape(num_layers, B * C, H) if h_in is not None else None
            
            self.gru.flatten_parameters()
            gru_out, h_out = self.gru(x_input, h_input)
            
            return gru_out.reshape(B, C, T, H).transpose(1, 2).contiguous(), h_out.reshape(num_layers, B, C, H).contiguous()
        
        gru_out_list, h_out_list = [], []
        for i in range(C):
            self.gru[i].flatten_parameters()
            h_input_i = h_in[:, :, i, :] if h_in is not None else None
            out_i, h_i = self.gru[i](x_emb[:, :, i, :], h_input_i)
            gru_out_list.append(out_i)
            h_out_list.append(h_i)
            
        return torch.stack(gru_out_list, dim=2).contiguous(), torch.stack(h_out_list, dim=2).contiguous()

    def predict_signals(self, hidden_states):
        """Pass hidden states through predictor(s). hidden_states shape: [B, T, C, H] or [B, C, H]"""
        C = self.model_cfg.input_size
        hidden_states = hidden_states.contiguous()
        
        if self.model_cfg.single_predictor:
            return self.predictor(hidden_states).squeeze(-1).contiguous()
        
        if hidden_states.dim() == 3:
            pred_list = [self.predictor[i](hidden_states[:, i, :].contiguous()) for i in range(C)]
            return torch.stack(pred_list, dim=1).squeeze(-1).contiguous()
            
        pred_list = [self.predictor[i](hidden_states[:, :, i, :].contiguous()) for i in range(C)]
        return torch.stack(pred_list, dim=2).squeeze(-1).contiguous()

    def forward(self, x): 
        B, C = x.size(0), self.model_cfg.input_size
        x = x.contiguous()
        
        x_emb = self.embed_signals(x)
        gru_out, _ = self.run_gru(x_emb)
        
        # Base prediction from t_0 to T_minus_1
        hidden_states = gru_out[:, :-1, :, :].contiguous()
        pred_curr = self.predict_signals(hidden_states)

        depth = self.model_cfg.loss.prediction_depth
        all_preds = [pred_curr]
        
        num_layers, H = self.model_cfg.rnn.num_layers, self.model_cfg.rnn.hidden_size
        T_minus_1 = hidden_states.size(1)
        
        # Broadcast h_curr for potential depth loop
        h_curr = hidden_states.reshape(B * T_minus_1, C, H).unsqueeze(0).repeat(num_layers, 1, 1, 1).contiguous()

        for d in range(1, depth):
            pred_flat = pred_curr.reshape(B * T_minus_1, 1, C)
            emb_curr = self.embed_signals(pred_flat)
            
            gru_out_step, h_curr = self.run_gru(emb_curr, h_in=h_curr)
            
            pred_curr_flat = self.predict_signals(gru_out_step.squeeze(1))
            pred_curr = pred_curr_flat.reshape(B, T_minus_1, C)
            
            # Truncate states exceeding timeline. E.g. t+2 cannot be forecasted at T-1
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
