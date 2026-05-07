import torch
from omegaconf import OmegaConf
from src.models.DECIFRA_MS import DECIFRA_MS
from src.models.saliency import compute_transfer_matrix_saliency

# Dummy config for testing
cfg_dict = {
    "single_embedder": True,
    "single_GRU": True,
    "rnn": {
        "input_embedding_size": 16,
        "hidden_size": 32,
    },
    "btp": {
        "hidden_dim": 32,
    },
    "single_predictor": True,
    "loss": {
        "threshold": 0.01,
        "sp_weight": 1.0,
        "forecast_weight": 1.0,
        "prediction_delay": 0,
        "prediction_depth": 1,
        "weighted_change": False,
    },
    "lr": 1e-3,
    "load_pretrained": False,
    "pretrained_path": None,
    "pretraining": True, 
    "input_size": 5, # 5 channels
    "output_size": 2,
    "n_training_stages": 2,
    "eps_stage_ratios": [0.3, 0.7],
    "transition_target_weight": 0.2
}
cfg = OmegaConf.create(cfg_dict)

model = DECIFRA_MS(cfg)

# Mock input data
B, T, C = 2, 10, 5 # batch size 2, 10 time points, 5 channels
x = torch.randn(B, T, C)

target_channel = 2

# Compute saliency
saliency_map = compute_transfer_matrix_saliency(
    model, x, target_channel=target_channel, baseline_type='identity', steps=10, delay=0, metric="mse"
)

print(f"Saliency map shape: {saliency_map.shape}")
print(f"Target channel: {target_channel}")

# Analyze row and column magnitudes
mean_abs_saliency = saliency_map.abs().mean(dim=(0, 1)) # shape: [C, C]

print("\nMean absolute saliency per row (how much each target channel's incoming connections matter):")
for i in range(C):
    print(f"Row {i}: {mean_abs_saliency[i].sum().item():.6f}")

print("\nMean absolute saliency per column (how much each source channel's outgoing connections matter):")
for j in range(C):
    print(f"Col {j}: {mean_abs_saliency[:, j].sum().item():.6f}")

print("\nSaliency matrix:")
print(mean_abs_saliency)

