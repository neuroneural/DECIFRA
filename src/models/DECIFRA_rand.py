import torch
from omegaconf import DictConfig
from src.models.DECIFRA_MS import DECIFRA_MS, BTP_MS_Gated_IMix_Res

def default_HPs(cfg: DictConfig):
    from src.models.DECIFRA_MS import default_HPs as ms_default
    return ms_default(cfg)

def custom_HPs(cfg: DictConfig, model_cfg_path: str):
    from src.models.DECIFRA_MS import custom_HPs as ms_custom
    return ms_custom(cfg, model_cfg_path)

class BTP_rand_Gated_IMix_Res(BTP_MS_Gated_IMix_Res):
    """
    Ablated version of BTP_MS_Gated_IMix_Res that replaces the
    data-driven query/key transition matrix with pure Gaussian noise.
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

class DECIFRA_rand(DECIFRA_MS):
    """
    Ablated MS version that generates a random noise transition matrix.
    Inherits all other attributes natively from DECIFRA_MS.
    """
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA_rand, self).__init__(model_cfg)
        
        # Override the BTP with the random ablation version
        self.BTP = BTP_rand_Gated_IMix_Res(
            input_dim=model_cfg.rnn.hidden_size, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=model_cfg.input_size
        )
