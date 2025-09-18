# base_model.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score,
)



from ..utils import zscore_np, corrcoef_batch

class BaseModel(nn.Module, ABC):
    """
    Base class for models in the codebase, sets the required API and provides
    default implementations for common functionality.

    Children may override:
      - handle_batch(self, batch)
      - get_optimizer(self, lr=None)
      - prepare_dataloader(data, labels, batch_size=64, shuffle=True)
    """

    def __init__(self, lr: float = None) -> None:
        super().__init__()
        self.lr = lr


    # ---- Defaults you may need to override for your models ----
    def compute_loss(
        self, loss_load: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Default cross entropy loss function used by many models.
        
        Returns
        -------
        loss : Tensor
            CE loss for backprop.
        logs : dict
            Loss dictionary for logs.
        """
        loss = cross_entropy(loss_load["logits"], targets)

        return loss, {"CE_loss": float(loss.detach().cpu().item())}


    def handle_batch(self, batch):
        """
        Standard batch handling routine for models. 
        Works for the majority of models that expect single input (time series or FNC).
        Only exception is FBNetGen that expects both time series and FNC.

        Returns
        -------
        loss : Tensor
            Loss for backprop (if you need it)
        batch_log : dict
            Dictionary of classification metrics and losses for logs.
        """
        # load batch into model
        
        data, labels = batch[:-1], batch[-1] # most of the time batch contains (data, labels), but sometimes there's more
        logits, loss_load = self.forward(*data)
        loss, loss_log = self.compute_loss(loss_load, labels)

        # compute metrics
        y_prob = softmax(logits, dim=1).detach().cpu().numpy()
        y_pred = y_prob.argmax(axis=1)
        y_true = labels.detach().cpu().numpy()

        batch_log = compute_metrics(y_prob, y_pred, y_true)
        batch_log.update(loss_log)

        return loss, batch_log

    def get_optimizer(self, lr: float | None = None):
        """ Basic Adam optimizer """
        if lr is None:
            lr = self.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    @staticmethod
    def prepare_dataloader(
        data, labels, type: str, shuffle: bool, batch_size: int = 64, zscore: bool = True
    ):
        """
        Dataloader factory for time series and FNC data.
        Creates a DataLoader from time series data and labels, derives FNC as PCC if needed.
        Most of the models hook their `prepare_dataloader` methods to this function.

        Args
        ----
        data : array-like, shape (B, T, D)
            Time series data of shape Batch x Time x (D)Features 
        labels : array-like, shape (B,)
            Class labels for the data.
        type : str
            Determines the type of data stored in the dataloader:
            "TS" for time series, "FNC" for connectivity matrices, "ALL" for both.
        shuffle : bool, optional
            Whether to shuffle batching in the DataLoader. Usually True for training, False for validation/test.
        batch_size : int, optional
            Batch size for the DataLoader.
        zscore : bool, optional, default=True
            Whether to z-score time series.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader generating the batches of data (as TS or FNC, or both) and labels.
        """

        if type == "TS" or type == "TS_only":
            if zscore:
                data = zscore_np(data, axis=1) # (B, T, D)
            data = torch.tensor(data, dtype=torch.float32)
            if type == "TS":
                labels = torch.tensor(labels, dtype=torch.int64)
                dataset = TensorDataset(data, labels)
            elif type == "TS_only":
                dataset = TensorDataset(data)

        elif type == "FNC":
            fnc = corrcoef_batch(data)  # (B, D, D)
            fnc = torch.tensor(fnc, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            dataset = TensorDataset(fnc, labels)
        elif type == "ALL":
            if zscore:
                data = zscore_np(data, axis=1) # (B, T, D)
            fnc = corrcoef_batch(data)  # (B, D, D)
            data = torch.tensor(data, dtype=torch.float32)
            fnc = torch.tensor(fnc, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            dataset = TensorDataset(data, fnc, labels)
        else:
            raise ValueError(f"Unknown data type: {type}. Supported types are 'TS', 'FNC', and 'ALL'.")
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def basic_ce_loss(logits, targets):
    """
    Default cross entropy loss function used by many models.
    
    Returns
    -------
    loss : Tensor
        CE loss for backprop.
    logs : dict
        Loss dictionary for logs.
    """
    loss = cross_entropy(logits, targets)

    return loss, {"CE_loss": float(loss.detach().cpu().item())}



def compute_metrics(y_prob, y_pred, y_true):
    """
    Compute a bundle of classification metrics.
    """
    log = {}
    log["accuracy"] = accuracy_score(y_true, y_pred)
    log["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    if y_prob.shape[1] == 2: # binary classification
        log["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    else: # multiclass classification
        log["auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    log["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    return log
