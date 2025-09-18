# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DECIFRA model """

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax

from omegaconf import OmegaConf, DictConfig

from .BaseModel import BaseModel, compute_metrics

def default_HPs(cfg: DictConfig):
    model_cfg = {
        "rnn": {
            "input_embedding_size": 16,
            "hidden_size": 16,
        },
        "btp": {
            "hidden_dim": 16,
        },
        "loss": {
            "threshold": 0.01,
            "sp_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
            "forecast_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
        },
        "lr": 1e-4,
        "load_pretrained": False,
        "pretrained_path": None,
        "input_size": cfg.data_info.feature_size,
        "output_size": cfg.data_info.n_classes,
    }
    return OmegaConf.create(model_cfg)


class DECIFRA(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA, self).__init__()

        self.model_cfg = model_cfg

        self.lr = model_cfg.lr
        self.pretraining = model_cfg.get("pretraining", False)

        # GRU and its input embedding
        self.embeddings = nn.Linear(1, model_cfg.rnn.input_embedding_size)
        self.gru = nn.GRU(model_cfg.rnn.input_embedding_size, model_cfg.rnn.hidden_size, batch_first=True)

        # bilinear transition predictor (BTP) used to compute the transfer matrices
        self.BTP = BTP(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

        # Classifier decoder
        self.clf = nn.Sequential(
            nn.Linear(model_cfg.input_size**2, model_cfg.input_size**2 // 2),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
            nn.Linear(model_cfg.input_size**2 // 2, model_cfg.input_size**2 // 4),
            nn.ReLU(),
            nn.Linear(model_cfg.input_size**2 // 4, model_cfg.output_size),
        )
        # Forecasting decoder
        self.predictor = nn.Linear(model_cfg.rnn.hidden_size, 1)

        self.criterion = DECIFRALoss(model_cfg)

    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def compute_loss(self, loss_load, targets):
        loss, log = self.criterion(loss_load, targets)

        return loss, log


    def handle_batch(self, batch):
        # load batch into model
        
        if self.pretraining:
            data = batch[0]
            _, loss_load = self.forward(*data)
            loss, batch_log = self.compute_loss(loss_load, None)

        else:
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
    
    def forward(self, x):
        B, T, C = x.shape  # [batch_size, time_length, input_size==self.input_size]
        orig_x = x

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, T, 1)
        embedded = self.embeddings(x) # shape: (B, C, T, self.model_cfg.rnn.input_embedding_size)

        # Initialize hidden state and run the recurrent loop
        h = torch.zeros(B, C, self.model_cfg.rnn.hidden_size, device=x.device)

        transfer_matrices = []
        hidden_states = []
        for t in range(T):
            # Process one time step
            gru_input = embedded[:, :, t, :].unsqueeze(2)  # (B, C, 1, embedding_dim)
            gru_input = gru_input.reshape(B*C, 1, self.model_cfg.rnn.input_embedding_size) # (B*C, 1, embedding_dim)
            # input hidden state must have shape (D * num_layers, N, hidden_size), D*num_layers = 1 in our case, N is effective GRU batch size
            h = h.reshape(1, B*C, self.model_cfg.rnn.hidden_size) # (1, B*C, hidden_dim)

            # update the hidden states with the new input by running GRU
            _, h = self.gru(gru_input, h) # output h shape is the same: (1, GRU_batch, hidden_size)
            h = h.reshape(B, C, self.model_cfg.rnn.hidden_size) # (B, C, hidden_dim)

            # Find transition matrix and mix hidden states
            h, mixing_matrix = self.BTP(h)
            hidden_states.append(h)
            transfer_matrices.append(mixing_matrix)

            if torch.any(torch.isnan(h)): # for debugging, h will have invalid values if model diverges
                raise Exception(f"h has nans at time point {t}")

        # Stack the alignment matrices, predict the next input 
        transfer_matrices = torch.stack(transfer_matrices, dim=1)  # (batch_size, seq_len, input_size, input_size)
        hidden_states = torch.stack(hidden_states, dim=1)[:, :-1, :, :] # brain latent states starting with time 0, [batch_size; time_length-1; input_size, hidden_dim]
        predicted = self.predictor(hidden_states).squeeze() # predictions of x starting with time 1, [batch_size; time_length-1; input_size]
        
        if self.pretraining:
            # pretrain on the forecasting task
            return None, {
                "matrices": transfer_matrices,
                "predicted": predicted,
                "originals": orig_x[:, 1:, :]
            }
        
        clf_input = transfer_matrices.reshape(B, T, -1) # [batch_size; time_length; input_size * input_size]
        time_logits = self.clf(clf_input) # [batch_size; time_length, n_classes]
        logits = torch.mean(time_logits, dim=1) # mean over time, [batch_size; n_classes]

        loss_load = {
            "logits": logits,
            "matrices": transfer_matrices,
            "time_logits": time_logits,
            "predicted": predicted,
            "originals": orig_x[:, 1:, :]
        }

        return logits, loss_load
    

class BTP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP, self).__init__()
        self.input_dim = input_dim

        self.gate = Gate(n_components)

        self.query = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.key = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        next_states = torch.bmm(transfer, x)

        return next_states, transfer

class Gate(nn.Module):
    def __init__(self, input_dim):
        super(Gate, self).__init__()
        self.bias = nn.Parameter(torch.randn(input_dim, input_dim))
    
    def forward(self, x):
        # Compute h_ij = abs(x_ij) + b_ij
        h = torch.abs(x) + self.bias
        
        # Compute a_ij = sigmoid(h_ij)
        a = torch.sigmoid(h)
        
        return a


class DECIFRALoss:
    """Forecasting + sparsity + (optional classification) loss for DECIFRA."""

    def __init__(self, model_cfg):
        self.sparsity_loss = InvertedHoyerMeasure(threshold=model_cfg.loss.threshold)

        self.sp_weight = model_cfg.loss.sp_weight
        self.forecast_weight = model_cfg.loss.forecast_weight


    def __call__(self, loss_load, targets):
        logits = loss_load["logits"] if "logits" in loss_load else None
        matrices = loss_load["matrices"]
        predicted = loss_load["predicted"]
        originals = loss_load["originals"]

        # Sparsity loss on the transfer matrices
        B, T, C, _ = matrices.shape
        matrices = matrices.reshape(B*T, C, C)
        sparse_loss = self.sparsity_loss(matrices)

        # Forecasting loss
        forecast_loss = F.mse_loss(predicted, originals)

        # Total loss
        loss = self.sp_weight * sparse_loss + self.forecast_weight * forecast_loss
        loss_components = {
            "sp_loss": sparse_loss.item(),
            "forecast_loss": forecast_loss.item(),
        }
        
        # Classification loss (if logits and targets are provided)
        if logits is not None and targets is not None: # training case
            ce_loss = F.cross_entropy(logits, targets)
            loss += ce_loss
            loss_components.update({
                "ce_loss": ce_loss.item(),
            })

        return loss, loss_components

class InvertedHoyerMeasure:
    """Sparsity loss function based on Hoyer measure: https://jmlr.csail.mit.edu/papers/volume5/hoyer04a/hoyer04a.pdf"""
    def __init__(self, 
                 threshold: float = 0.01,
                 eps: float = 1e-12,
                 ):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, x):
        B, C, C = x.shape
        # Assuming x has shape (batch_size, input_dim, input_dim)        
        n = x[0].numel()
        sqrt_n = torch.sqrt(torch.tensor(float(n), device=x.device))
        assert n == C * C, f"Expected square matrices, got {x.shape}"

        v = x.view(B, -1)
        l1 = v.abs().sum(dim=1)
        l2 = torch.linalg.vector_norm(v, ord=2, dim=1).clamp_min(self.eps)

        numerator = sqrt_n - l1 / l2
        denominator = sqrt_n - 1
        mod_hoyer = 1 - (numerator / denominator) # = 0 if perfectly sparse, 1 if all are equal

        z = mod_hoyer - self.threshold
        loss = F.leaky_relu(z)

        mean_loss = torch.mean(loss)

        return mean_loss