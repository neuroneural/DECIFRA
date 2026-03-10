# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" LSTM model for fMRI forecasting, used as a baseline """

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax

from omegaconf import OmegaConf, DictConfig

from src.models.BaseModel import BaseModel, compute_metrics
from torch.nn.functional import cross_entropy, mse_loss

def default_HPs(cfg: DictConfig):
    model_cfg = { 
        "input_size": cfg.data_info.feature_size,
        "output_size": cfg.data_info.n_classes,
    }
    return OmegaConf.create(model_cfg)


class LSTM(BaseModel):
    def __init__(
            self,
            model_cfg,
            hidden_size: int = 210,
            num_layers: int = 2,
            bidirectional: bool = False,
            dropout: float = 0.5,
            lr: float = 5e-4,
    ):
        super().__init__()

        self.lr = lr

        input_size = model_cfg.input_size

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = 2 * hidden_size if bidirectional else hidden_size

        self.predictor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                lstm_output_size,
                lstm_output_size,
            ),
            nn.ReLU(),
            nn.Linear(
                lstm_output_size,
                input_size,
            ),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        preds = self.predictor(lstm_output[:, :-1, :])
        origs = x[:, 1:, :]

        return None, {"preds": preds, "origs": origs}

    #### Helper functions for model training and evaluation ####

    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)
    
    def compute_loss(self, loss_load, targets):
        """
        Standard loss computation routine for models.
        Args:
            loss_load (dict): Forward's second output for the batch.
            targets (torch.Tensor): True labels for the batch.

        Returns
        -------
        loss : Tensor
            Loss for backpropagation.
        logs : dict
            Dictionary containing the loss components for logs.
        """
        loss = mse_loss(loss_load["preds"], loss_load["origs"])

        return loss, {"forecast_loss": float(loss.detach().cpu().item())}

    def handle_batch(self, batch):
        # load batch into model
        
        _, loss_load = self.forward(*batch)
        loss, batch_log = self.compute_loss(loss_load, None)


        return loss, batch_log
    
