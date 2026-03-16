# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DECIFRA model """

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax

from omegaconf import OmegaConf, DictConfig

from src.models.BaseModel import BaseModel, compute_metrics

def default_HPs(cfg: DictConfig):
    model_cfg = {
        "single_embedder": True, # TODO: implement if True, use the same embedding layer for all input channels; if False, one embedding layer per channel
        "single_GRU": True, # TODO: implement if True, use the same GRU for all input channels; if False, one GRU per channel
        "rnn": {
            "input_embedding_size": 16,
            "hidden_size": 32,
        },
        "btp": {
            "hidden_dim": 32,
        },
        "single_predictor": True, # TODO: implement if True, use one predictor for all input channels; if False, one predictor per channel
        
        "loss": {
            "threshold": 0.01,
            "sp_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
            "forecast_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
            "prediction_delay": 0, # given input time series with points 0...T, predict points starting from time 1+prediction_delay
            "prediction_depth": 1, # how deep the forcaster forecasts: 1 predicts only next signal, 2 reiterates on the previous prediction...
            "weighted_change": False, # if True, the loss will penalize forecasting errors on high-amplitude changes more
        },
        "lr": 1e-3,
        "load_pretrained": False,
        "pretrained_path": None,
        "pretraining": True, 
        "input_size": cfg.data_info.feature_size,
        "output_size": cfg.data_info.n_classes,
    }
    return OmegaConf.create(model_cfg)


def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.input_size = cfg.data_info.feature_size
    model_cfg.output_size = cfg.data_info.n_classes
    return model_cfg


class DECIFRA(BaseModel):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA, self).__init__()

        self.model_cfg = model_cfg

        self.lr = model_cfg.lr
        self.pretraining = model_cfg.get("pretraining", False)

        # GRU and its input embedder
        if model_cfg.single_embedder:
            self.embedder = nn.Linear(1, model_cfg.rnn.input_embedding_size)
        if model_cfg.single_GRU:
            self.gru = nn.GRU(model_cfg.rnn.input_embedding_size, model_cfg.rnn.hidden_size, batch_first=True)

        # bilinear transition predictor (BTP) used to compute the transfer matrices
        self.BTP = BTP(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

        # Forecasting decoder
        if model_cfg.single_predictor:
            self.predictor = nn.Linear(model_cfg.rnn.hidden_size, 1)

        if not self.pretraining:
            # Classifier decoder
            self.clf = nn.Sequential(
                nn.Linear(model_cfg.input_size**2, model_cfg.input_size**2 // 2),
                nn.ReLU(),
                nn.Dropout1d(p=0.3),
                nn.Linear(model_cfg.input_size**2 // 2, model_cfg.input_size**2 // 4),
                nn.ReLU(),
                nn.Linear(model_cfg.input_size**2 // 4, model_cfg.output_size),
            )

    @staticmethod
    def prepare_dataloader(data, labels, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, labels, "TS", shuffle, batch_size, zscore)
    
    @staticmethod
    def prepare_pretraining_dataloader(data, shuffle: bool, batch_size: int = 64, zscore: bool = True):
        return BaseModel.prepare_dataloader(data, None, "TS_only", shuffle, batch_size, zscore)

    def compute_loss(self, loss_load, targets):
        loss, log = decifra_loss(self.model_cfg.loss, self.pretraining, loss_load, targets)

        return loss, log


    def handle_batch(self, batch):
        # load batch into model
        
        if self.pretraining:
            _, loss_load = self.forward(*batch)
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
    
    ### actual functions for the model's forward pass
    def embed_signals(self, x):
        # B, T, C = x.shape  # [batch_size, time_length, input_size==self.input_size]
        if self.model_cfg.single_embedder:
            return self.embedder(x.unsqueeze(-1))

    def recurrent_step(self, x, h):
        pass

    def run_recurrent_loop(self, x, h):
        pass

    def forward(self, x): 
        B, T, C = x.shape  # [batch_size, time_length, input_size==self.input_size]
        orig_x = x

        E = self.model_cfg.rnn.input_embedding_size
        H = self.model_cfg.rnn.hidden_size

        # 1) Embed all input signals
        x_emb = self.embed_signals(x)  # [B, T, C, E]

        # 2) Run recurrent loop
        h = torch.zeros(B, C, H, device=x.device)

        transfer_matrices = []
        hidden_states = []
        for t in range(T):
            # Process one time step

            # prepare GRU input
            x_input = x_emb[:, t, :, :].reshape(B*C, 1, E) # [B*C, 1, E] = [effective_GRU_batch, seq_len=1, GRU_input_dim]
            h_input = h.reshape(1, B*C, H) # [1, B*C, hidden_dim] = [bidirectional_GRU*GRU_num_layers = 1, effective_GRU_batch, hidden_size]

            if self.model_cfg.single_GRU:
                _, new_h = self.gru(x_input, h_input) # output h shape is the same: (1, GRU_batch, hidden_size)

            h = new_h.reshape(B, C, H) # (B, C, H)

            # Derive transition matrix and mix hidden states
            h, mixing_matrix = self.BTP(h)

            # save outputs
            hidden_states.append(h)
            transfer_matrices.append(mixing_matrix)

            if torch.any(torch.isnan(h)): # for debugging, h will have invalid values if model diverges
                raise Exception(f"h has nans at time point {t}")

        # Stack the transition matrices, predict the next input
        transfer_matrices = torch.stack(transfer_matrices, dim=1)  # (batch_size, seq_len, input_size, input_size)
        hidden_states = torch.stack(hidden_states, dim=1)[:, :-1, :, :] # brain latent states starting with time 0, [batch_size; time_length-1; input_size, hidden_dim]
        predicted = self.predictor(hidden_states).squeeze() # predictions of x starting with time 1, [batch_size; time_length-1; input_size]
        
        if self.pretraining:
            # pretrain on the forecasting task
            return None, {
                "matrices": transfer_matrices,
                "predicted": predicted,
                "originals": orig_x,
            }
        
        clf_input = transfer_matrices.reshape(B, T, -1) # [batch_size; time_length; input_size * input_size]
        time_logits = self.clf(clf_input) # [batch_size; time_length, n_classes]
        logits = torch.mean(time_logits, dim=1) # mean over time, [batch_size; n_classes]

        loss_load = {
            "logits": logits,
            "matrices": transfer_matrices,
            "time_logits": time_logits,
            "predicted": predicted,
            "originals": orig_x,
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


def decifra_loss(loss_cfg, pretraining, loss_load, targets=None):
    """Forecasting + sparsity + (optional classification) loss for DECIFRA."""
    sp_weight = loss_cfg.sp_weight
    forecast_weight = loss_cfg.forecast_weight
    threshold = loss_cfg.threshold

    delay = loss_cfg.prediction_delay
    # depth = loss_cfg.prediction_depth
    # weighted_change = loss_cfg.weighted_change

    matrices = loss_load["matrices"]

    originals = loss_load["originals"]
    target_signal = originals[:, 1+delay:, :]
    predicted_signal = loss_load["predicted"][:, delay:, :]
    dumb_predicted_signal = originals[:, delay:-1, :]

    # Sparsity loss on the transfer matrices
    B, T, C, _ = matrices.shape
    matrices = matrices.reshape(B*T, C, C)
    sparse_loss = inverted_hoyer_measure(matrices, threshold=threshold)
 
    # Forecasting loss
    forecast_loss = F.mse_loss(predicted_signal, target_signal)
    dumb_forecast_loss = F.mse_loss(dumb_predicted_signal, target_signal)

    # Total loss
    loss = sp_weight * sparse_loss + forecast_weight * forecast_loss
    loss_components = {
        "sp_loss": sparse_loss.item(),
        "forecast_loss": forecast_loss.item(),
        "dumb_forecast_loss": dumb_forecast_loss.item(),
    }

    if not pretraining:
        logits = loss_load["logits"] if "logits" in loss_load else None
        assert logits is not None and targets is not None, "In classification mode, both logits and targets must be provided to compute the classification loss."
        
        ce_loss = F.cross_entropy(logits, targets)
        loss += ce_loss
        loss_components.update({
            "ce_loss": ce_loss.item(),
        })

    return loss, loss_components

def inverted_hoyer_measure(x, 
                         threshold,
                         eps: float = 1e-12
                         ):
    """Sparsity loss function based on Hoyer measure: https://jmlr.csail.mit.edu/papers/volume5/hoyer04a/hoyer04a.pdf"""

    B, C, C = x.shape
    # Assuming x has shape (batch_size, input_dim, input_dim)        
    n = x[0].numel()
    sqrt_n = torch.sqrt(torch.tensor(float(n), device=x.device))
    assert n == C * C, f"Expected square matrices, got {x.shape}"

    v = x.view(B, -1)
    l1 = v.abs().sum(dim=1)
    l2 = torch.linalg.vector_norm(v, ord=2, dim=1).clamp_min(eps)

    numerator = sqrt_n - l1 / l2
    denominator = sqrt_n - 1
    mod_hoyer = 1 - (numerator / denominator) # = 0 if perfectly sparse, 1 if all are equal

    z = mod_hoyer - threshold
    loss = F.leaky_relu(z)

    mean_loss = torch.mean(loss)

    return mean_loss

##### Variations of the model
## no gate

# DECIFRA with no gate
class BTP_noGate(BTP):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_noGate, self).__init__(input_dim, hidden_dim, n_components)

    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        # gate = self.gate(transfer)
        # transfer = transfer * gate

        next_states = torch.bmm(transfer, x)

        return next_states, transfer
    
class DECIFRA_noGate(DECIFRA):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_noGate, self).__init__(model_cfg)

        # overwrite the BTP with the no-gate version
        self.BTP = BTP_noGate(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

# DECIFRA with no gate and identity mixing + learned residual connections

class BTP_noGate_IMix_Res(BTP):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_noGate_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        self.res_weight = nn.Parameter(torch.randn(1))

    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        n_components = x.size(1)

        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        # gate = self.gate(transfer)
        # transfer = transfer * gate

        identity = torch.eye(n_components, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1) # shape (batch_size, input_dim, input_dim)
        full_transfer = transfer + identity
        next_states = torch.bmm(full_transfer, x)

        return next_states, transfer

class DECIFRA_noGate_IMix_Res(DECIFRA):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_noGate_IMix_Res, self).__init__(model_cfg)

        # overwrite the BTP with the no-gate version and identity mixing + residuals
        self.BTP = BTP_noGate_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

## with gate

# DECIFRA with no mixing of hidden states (identity transfer matrix)
class BTP_IMix(BTP):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_IMix, self).__init__(input_dim, hidden_dim, n_components)

    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        next_states = x

        return next_states, transfer
    
class DECIFRA_IMix(DECIFRA):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_IMix, self).__init__(model_cfg)

        # overwrite the BTP with the no-gate version and identity mixing
        self.BTP = BTP_IMix(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

# DECIFRA with identity mixing + learned residual connections
class BTP_IMix_Res(BTP):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        self.res_weight = nn.Parameter(torch.randn(1))

    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        n_components = x.size(1)

        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        identity = torch.eye(n_components, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1) # shape (batch_size, input_dim, input_dim)
        full_transfer = transfer + identity
        next_states = torch.bmm(full_transfer, x)

        return next_states, transfer

class DECIFRA_IMix_Res(DECIFRA):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_IMix_Res, self).__init__(model_cfg)

        # overwrite the BTP with the version with identity mixing and residuals
        self.BTP = BTP_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )

# DECIFRA with identity mixing + learned residual connections
class BTP_Gated_IMix_Res(BTP):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_Gated_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        self.res_weight = nn.Parameter(torch.randn(1))

    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        n_components = x.size(1)

        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        identity = torch.eye(n_components, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1) # shape (batch_size, input_dim, input_dim)
        full_transfer = transfer + identity
        gate = self.gate(full_transfer)
        full_transfer = full_transfer * gate
        next_states = torch.bmm(full_transfer, x)

        return next_states, full_transfer

class DECIFRA_Gated_IMix_Res(DECIFRA):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_Gated_IMix_Res, self).__init__(model_cfg)

        # overwrite the BTP with the version with identity mixing and residuals
        self.BTP = BTP_Gated_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )