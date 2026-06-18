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
