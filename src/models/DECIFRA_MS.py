import torch
from torch import nn
from omegaconf import OmegaConf, DictConfig
from src.models.DECIFRA import DECIFRA, BTP, Gate, default_HPs as vanilla_HPs

def default_HPs(cfg: DictConfig):
    model_cfg = vanilla_HPs(cfg)
    model_cfg.n_training_stages = 2
    model_cfg.eps_stage_ratios = [0.3, 0.7]
    model_cfg.transition_target_weight = 0.2
    return model_cfg


def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    from src.models.DECIFRA import custom_HPs as vanilla_custom_HPs
    model_cfg = vanilla_custom_HPs(cfg, model_cfg_path)
    return model_cfg

class BTP_MS_Gated_IMix_Res(BTP):
    """
    Multistage-aware BTP variant: Gated + Identity Mixing + Residual Connections.
    Supports off-diagonal scaling of the dynamic transfer part.
    """
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_MS_Gated_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        self.transfer_weight = 1.0



    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        # Multistage Scaling: Scale the dynamic part
        if self.transfer_weight != 1.0:
            transfer = transfer * self.transfer_weight

        if self.transfer_weight != 0.0:
            next_states = x + torch.bmm(transfer, x)
        else:
            next_states = x

        # Return next_states and transfer (without identity to save memory inside the loop)
        return next_states, transfer

class DECIFRA_MS(DECIFRA):
    """
    Multistage version of DECIFRA that supports dynamic unfreezing of cross-channel 
    parameters across multiple training stages.
    """
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_MS, self).__init__(model_cfg)
        
        # Override the BTP with the Gated IMix Res MS-aware version
        self.BTP = BTP_MS_Gated_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )
        
        # Initialize scaling parameters
        self.current_stage = 0
        self.total_stages = model_cfg.n_training_stages
        self.stage_epochs = 0
        self.current_scale = 1.0
        self.transition_target_weight = model_cfg.transition_target_weight

    def forward(self, x):
        logits, loss_load = super().forward(x)
        # Vectorized addition of identity matrix to transfer matrices to optimize memory
        matrices = loss_load["matrices"]  # (B, T, C, C)
        C = matrices.size(-1)
        identity = torch.eye(C, device=matrices.device).view(1, 1, C, C)
        loss_load["matrices"] = matrices + identity
        return logits, loss_load

    def set_stage(self, stage_idx, num_epochs):
        """Called by StagePreTrainer when a new stage begins."""
        self.current_stage = stage_idx
        self.stage_epochs = num_epochs

    def set_epoch(self, epoch_idx):
        """Called by Trainer at the start of every epoch."""
        # Stage-specific scaling logic
        if self.total_stages < 2:
            self.current_scale = self.transition_target_weight
        else:
            if self.current_stage == 0:
                self.current_scale = 0.0
            elif self.current_stage == 1:
                # Linear interpolation from start to end
                start, end = 0.0, self.transition_target_weight
                progress = epoch_idx / (self.stage_epochs - 1 if self.stage_epochs > 1 else 1)
                self.current_scale = start + progress * (end - start)
            else:
                self.current_scale = self.transition_target_weight
        
        self.BTP.transfer_weight = self.current_scale

    def handle_batch(self, batch):
        loss, log = super().handle_batch(batch)
        log["transfer_weight"] = self.current_scale
        return loss, log


class BTP_rand_Gated_IMix_Res(BTP_MS_Gated_IMix_Res):
    """
    Ablated version of BTP_MS_Gated_IMix_Res that replaces the data-driven
    query/key transition matrix with pure Gaussian noise.
    """
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_rand_Gated_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        # Remove queries and keys parameters since they are unused
        del self.query
        del self.key

    def forward(self, x):
        n_components = x.size(1)

        # ABLATION: Generate Gaussian noise transfer matrix instead of relying on sequence data
        transfer = torch.randn(x.size(0), n_components, n_components, device=x.device)

        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        # Multistage Scaling: Scale the dynamic off-diagonal part
        if self.transfer_weight != 1.0:
            transfer = transfer * self.transfer_weight

        if self.transfer_weight != 0.0:
            next_states = x + torch.bmm(transfer, x)
        else:
            next_states = x

        # add identity to the output transfer matrix
        identity = torch.eye(n_components, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        transfer = transfer + identity

        return next_states, transfer


class DECIFRA_MS_rand(DECIFRA_MS):
    """
    Ablated MS variant that generates a random noise transition matrix.
    Inherits all other behaviour (staging, forward, loss) from DECIFRA_MS.
    Select via: model=DECIFRA_MS/<cfg> model.variant=rand
    """
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_MS_rand, self).__init__(model_cfg)

        # Override the BTP with the random ablation version
        self.BTP = BTP_rand_Gated_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size,
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )