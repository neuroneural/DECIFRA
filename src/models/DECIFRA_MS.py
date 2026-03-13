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

class BTP_MS_Gated_IMix_Res(BTP):
    """
    Multistage-aware BTP variant: Gated + Identity Mixing + Residual Connections.
    Supports off-diagonal scaling of the dynamic transfer part.
    """
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP_MS_Gated_IMix_Res, self).__init__(input_dim, hidden_dim, n_components)
        self.off_diag_scale = 1.0

    def forward(self, x):
        n_components = x.size(1)

        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        # Multistage Scaling: Scale off-diagonal of the dynamic part
        if self.off_diag_scale != 1.0:
            mask = torch.eye(n_components, device=x.device).unsqueeze(0)
            transfer = transfer * mask + (transfer * (1 - mask)) * self.off_diag_scale

        identity = torch.eye(n_components, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        full_transfer = transfer + identity
        
        gate = self.gate(full_transfer)
        full_transfer = full_transfer * gate
        next_states = torch.bmm(full_transfer, x)

        return next_states, full_transfer

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
        self.total_stages = model_cfg.get("n_training_stages", 1)
        self.stage_epochs = 0
        self.current_scale = 1.0
        self.transition_target_weight = model_cfg.transition_target_weight

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
        
        self.BTP.off_diag_scale = self.current_scale

    def handle_batch(self, batch):
        loss, log = super().handle_batch(batch)
        log["off_diag_scale"] = self.current_scale
        return loss, log