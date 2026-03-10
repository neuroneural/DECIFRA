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

    def forward(self, x): 
        # x shape: [B, T, C]
        B, T, C = x.shape
        orig_x = x

        E = self.model_cfg.rnn.input_embedding_size
        H = self.model_cfg.rnn.hidden_size

        # 1. Embed signals
        if self.model_cfg.single_embedder:
            x_emb = self.embedder(x.unsqueeze(-1))  # [B, T, C, E]
        else:
            x_emb_list = [self.embedder[i](x[:, :, i].unsqueeze(-1)) for i in range(C)]
            x_emb = torch.stack(x_emb_list, dim=2)  # [B, T, C, E]

        # 2. Run GRU (force teaching over the entire sequence T)
        if self.model_cfg.single_GRU:
            # Process all channels effectively in parallel treating them as independent items in batch
            x_input = x_emb.transpose(1, 2).reshape(B * C, T, E)  # [B*C, T, E]
            gru_out, _ = self.gru(x_input)  # [B*C, T, H]
            gru_out = gru_out.reshape(B, C, T, H).transpose(1, 2)  # [B, T, C, H]
        else:
            # Process each channel independently with its own GRU
            gru_out_list = []
            for i in range(C):
                x_input_i = x_emb[:, :, i, :]  # [B, T, E]
                out_i, _ = self.gru[i](x_input_i)  # [B, T, H]
                gru_out_list.append(out_i)
            gru_out = torch.stack(gru_out_list, dim=2)  # [B, T, C, H]

        # 3. Forecast prediction (Depth-based Autoregressive Loop)
        # We predict using the hidden state from the previous time step.
        # hidden_states represent the state after seeing x_0, x_1, ... x_t
        hidden_states = gru_out[:, :-1, :, :]  # [B, T-1, C, H]
        
        # Initial depth 1 prediction (predicts x_{t+1})
        if self.model_cfg.single_predictor:
            pred_curr = self.predictor(hidden_states).squeeze(-1) # [B, T-1, C]
        else:
            predicted_list = [self.predictor[i](hidden_states[:, :, i, :]) for i in range(C)]
            pred_curr = torch.stack(predicted_list, dim=2).squeeze(-1) # [B, T-1, C]

        depth = self.model_cfg.loss.get("prediction_depth", 1)
        all_preds = [pred_curr]
        
        h_curr = hidden_states

        # Optional prediction depth loop
        for d in range(1, depth):
            # 1. Embed current predictions
            if self.model_cfg.single_embedder:
                emb_curr = self.embedder(pred_curr.unsqueeze(-1)) # [B, T-1, C, E]
            else:
                emb_curr_list = [self.embedder[i](pred_curr[:, :, i].unsqueeze(-1)) for i in range(C)]
                emb_curr = torch.stack(emb_curr_list, dim=2) # [B, T-1, C, E]
            
            # 2. Run 1 step of GRU to update h_curr to h_next
            if self.model_cfg.single_GRU:
                x_input = emb_curr.reshape(B * (T-1) * C, 1, E) 
                # h_input needs to be [num_layers, batch_size, H]
                num_layers = self.model_cfg.rnn.get("num_layers", 1)
                h_input = h_curr.reshape(B * (T-1) * C, H).unsqueeze(0).repeat(num_layers, 1, 1) # [L, B*(T-1)*C, H]
                
                _, h_next = self.gru(x_input, h_input) # h_next is [L, B*(T-1)*C, H]
                h_curr = h_next[-1].reshape(B, T-1, C, H) # take the last layer hidden state
            else:
                h_next_list = []
                for i in range(C):
                    x_input_i = emb_curr[:, :, i, :].reshape(B * (T-1), 1, E) 
                    
                    num_layers = self.model_cfg.rnn.get("num_layers", 1)
                    h_input_i = h_curr[:, :, i, :].reshape(B * (T-1), H).unsqueeze(0).repeat(num_layers, 1, 1)

                    _, h_next_i = self.gru[i](x_input_i, h_input_i)
                    h_next_list.append(h_next_i[-1].reshape(B, T-1, H))
                h_curr = torch.stack(h_next_list, dim=2) 

            # 3. Predict next step
            if self.model_cfg.single_predictor:
                pred_curr = self.predictor(h_curr).squeeze(-1) # [B, T-1, C]
            else:
                pred_curr_list = [self.predictor[i](h_curr[:, :, i, :]) for i in range(C)]
                pred_curr = torch.stack(pred_curr_list, dim=2).squeeze(-1) # [B, T-1, C]

            all_preds.append(pred_curr)

        # Output predictions: shape [B, T-1, C, depth]
        predicted = torch.stack(all_preds, dim=-1)

        loss_load = {
            "predicted": predicted,
            "originals": orig_x,
        }

        # As a purely forecasting model, we return logits=None
        return None, loss_load

    def compute_loss(self, loss_load, targets):
        delay = self.model_cfg.loss.get("prediction_delay", 0)
        depth = self.model_cfg.loss.get("prediction_depth", 1)
        forecast_weight = self.model_cfg.loss.forecast_weight

        originals = loss_load["originals"]
        predicted = loss_load["predicted"] # [B, T-1, C, depth]

        T = originals.size(1)

        total_forecast_loss = 0.0
        
        # We calculate MSE over each predicted depth step
        # depth=0 is t+1, depth=1 is t+2, etc. (with delay added on top)
        for d in range(depth):
            total_shift = delay + d
            
            # target sequence starts at 1 + total_shift
            # If shift is too large, we break so we don't index out of bounds
            if 1 + total_shift >= T:
                break
                
            target_signal = originals[:, 1 + total_shift:, :]
            
            # predicted[..., d] corresponds to predicting target_signal
            # predicted[..., d] has T-1 time points (predictions for 0 to T-2)
            # but we only evaluate the overlapping portion.
            # prediction at 0 corresponds to t=1 + d
            # prediction at T-2-total_shift corresponds to t=T-1
            end_idx = (T - 1) - total_shift
            pred_d = predicted[:, delay:delay+end_idx, :, d] 
            
            total_forecast_loss += mse_loss(pred_d, target_signal)

        # Average over whatever depths were successfully matched
        total_forecast_loss = total_forecast_loss / depth

        loss = forecast_weight * total_forecast_loss
        loss_components = {
            "forecast_loss": total_forecast_loss.item(),
        }

        return loss, loss_components

    def handle_batch(self, batch):
        # Even if batch has labels, we only extract data for purely forecasting models
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        _, loss_load = self.forward(data)
        loss, batch_log = self.compute_loss(loss_load, None)

        return loss, batch_log
