# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DECIFRA model """

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax, cross_entropy, mse_loss

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
            "bidirectional": False,
        },
        "single_predictor": True, 
        
        "loss": {
            "prediction_delay": 0,
            "forecast_weight": 1.0, 
            "classification_weight": 0.02, # used when pretraining is False
        },
        "lr": 1e-3,
        "load_pretrained": False,
        "pretrained_path": None,
        "pretraining": True, 
        "input_size": cfg.data_info.feature_size,
        "output_size": cfg.data_info.n_classes,
    }
    return OmegaConf.create(model_cfg)


class meanGRU(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(meanGRU, self).__init__()

        self.model_cfg = model_cfg
        self.lr = model_cfg.lr
        self.pretraining = model_cfg.get("pretraining", False)
        
        C = model_cfg.input_size
        E = model_cfg.rnn.input_embedding_size
        H = model_cfg.rnn.hidden_size
        num_layers = model_cfg.rnn.get("num_layers", 1)
        dropout = model_cfg.rnn.get("dropout", 0.0)
        bidirectional = model_cfg.rnn.get("bidirectional", False)
        D = 2 if bidirectional else 1
        
        # 1. Embedder
        if model_cfg.single_embedder:
            self.embedder = nn.Linear(1, E)
        else:
            self.embedder = nn.ModuleList([nn.Linear(1, E) for _ in range(C)])

        # 2. GRU
        if model_cfg.single_GRU:
            self.gru = nn.GRU(
                E, H, batch_first=True, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            self.gru = nn.ModuleList([
                nn.GRU(
                    E, H, batch_first=True, num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=bidirectional
                ) for _ in range(C)
            ])

        # 3. Predictor (forecasting)
        if model_cfg.single_predictor:
            self.predictor = nn.Linear(H * D, 1)
        else:
            self.predictor = nn.ModuleList([nn.Linear(H * D, 1) for _ in range(C)])

        # 4. Classifier
        if not self.pretraining:
            # We will mean-pool the outputs over time for each channel, 
            # then concatenate the results: shape will be C * H * D
            self.clf = nn.Sequential(
                nn.Linear(C * H * D, (C * H * D) // 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear((C * H * D) // 2, (C * H * D) // 4),
                nn.ReLU(),
                nn.Linear((C * H * D) // 4, model_cfg.output_size),
            )

    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def forward(self, x): 
        # x shape: [B, T, C]
        B, T, C = x.shape
        orig_x = x

        E = self.model_cfg.rnn.input_embedding_size

        # 1. Embed signals
        if self.model_cfg.single_embedder:
            x_emb = self.embedder(x.unsqueeze(-1))  # [B, T, C, E]
        else:
            x_emb_list = [self.embedder[i](x[:, :, i].unsqueeze(-1)) for i in range(C)]
            x_emb = torch.stack(x_emb_list, dim=2)  # [B, T, C, E]

        # 2. Run GRU
        if self.model_cfg.single_GRU:
            # Process all channels effectively in parallel treating them as independent items in batch
            x_input = x_emb.transpose(1, 2).reshape(B * C, T, E)  # [B*C, T, E]
            gru_out, _ = self.gru(x_input)  # [B*C, T, H*D]
            gru_out = gru_out.reshape(B, C, T, -1).transpose(1, 2)  # [B, T, C, H*D]
        else:
            # Process each channel independently with its own GRU
            gru_out_list = []
            for i in range(C):
                x_input_i = x_emb[:, :, i, :]  # [B, T, E]
                out_i, _ = self.gru[i](x_input_i)  # [B, T, H*D]
                gru_out_list.append(out_i)
            gru_out = torch.stack(gru_out_list, dim=2)  # [B, T, C, H*D]

        # 3. Forecast prediction
        # We predict using the hidden state from the previous time step.
        # So we take gru_out[:, :-1, :, :] to predict x[:, 1:, :] 
        # (Actually, we just output `predicted` for all seq_len-1 steps)
        # We need `hidden_states` [B, T-1, C, H*D] for the predictor
        hidden_states = gru_out[:, :-1, :, :]  

        if self.model_cfg.single_predictor:
            predicted = self.predictor(hidden_states).squeeze(-1) # [B, T-1, C]
        else:
            predicted_list = [self.predictor[i](hidden_states[:, :, i, :]) for i in range(C)]
            predicted = torch.stack(predicted_list, dim=2).squeeze(-1) # [B, T-1, C]

        # 4. Handle pretraining / classification
        if self.pretraining:
            return None, {
                "predicted": predicted,
                "originals": orig_x,
            }
        
        # Classification path: pool over time for each channel, then classify
        # mean pool over time dimension T
        time_pooled = torch.mean(gru_out, dim=1)  # [B, C, H*D]
        clf_input = time_pooled.reshape(B, -1)  # [B, C * H * D]
        logits = self.clf(clf_input)  # [B, n_classes]

        loss_load = {
            "logits": logits,
            "predicted": predicted,
            "originals": orig_x,
        }

        return logits, loss_load

    def compute_loss(self, loss_load, targets):
        delay = self.model_cfg.loss.get("prediction_delay", 0)
        forecast_weight = self.model_cfg.loss.forecast_weight

        originals = loss_load["originals"]
        predicted = loss_load["predicted"]

        # target_signal starts at 1 + delay
        target_signal = originals[:, 1+delay:, :]
        predicted_signal = predicted[:, delay:, :]

        forecast_loss = mse_loss(predicted_signal, target_signal)

        loss = forecast_weight * forecast_loss
        loss_components = {
            "forecast_loss": forecast_loss.item(),
        }

        if not self.pretraining:
            logits = loss_load["logits"]
            assert logits is not None and targets is not None, "In classification mode, both logits and targets must be provided."
            
            clf_weight = self.model_cfg.loss.get("classification_weight", 0.02)
            ce_loss = cross_entropy(logits, targets)
            loss += clf_weight * ce_loss
            loss_components.update({
                "ce_loss": ce_loss.item(),
            })

        return loss, loss_components

    def handle_batch(self, batch):
        if self.pretraining:
            # During pretraining, batch is just (data,) or (data, label)
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            _, loss_load = self.forward(data)
            loss, batch_log = self.compute_loss(loss_load, None)
        else:
            data, labels = batch[:-1], batch[-1]
            logits, loss_load = self.forward(*data)
            loss, loss_log = self.compute_loss(loss_load, labels)

            # Compute metrics
            y_prob = softmax(logits, dim=1).detach().cpu().numpy()
            y_pred = y_prob.argmax(axis=1)
            y_true = labels.detach().cpu().numpy()

            batch_log = compute_metrics(y_prob, y_pred, y_true)
            batch_log.update(loss_log)

        return loss, batch_log
