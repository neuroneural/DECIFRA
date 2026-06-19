import numpy as np
import torch

def load_dummy_data(n_samples=1000, seq_len=200, n_channels=50):
    """
    Generates a dummy dataset for testing scripts and memory utilization limits. 
    Returns: torch.Tensor of shape (n_samples, seq_len, n_channels)
    """
    # Generating standard normal distributed data
    data = torch.randn(n_samples, seq_len, n_channels)
    return data


def load_pretraining_data(ds_cfg=None):
    """Uniform pretraining entrypoint (see src.registry)."""
    return load_dummy_data()


def load_finetuning_data(ds_cfg=None):
    """Random labelled data for smoke-testing the fine-tuning pipeline."""
    n_classes = int(ds_cfg.get("n_classes", 2)) if ds_cfg is not None else 2
    data = load_dummy_data().numpy()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, n_classes, size=data.shape[0])
    return data, labels
