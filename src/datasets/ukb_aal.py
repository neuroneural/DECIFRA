"""
UKB AAL dataset (AAL atlas ROI time series), consolidated across all variants.

Variants (selected via the dataset config's `variant` field):
  - hold : holdout split   (data_holdout.npy)
  - 2000 : 2000 subjects   (data_2000.npy)
  - 4000 : 4000 subjects   (data_4000.npy)

`load_pretraining_data(ds_cfg)` is the registry entrypoint. `load_aal(variant)`
returns the raw array. `load_data_hold` is kept as a back-compat alias.
"""
import os

import numpy as np

from src.settings import DATA_ROOT

AAL_DIR = os.path.join(DATA_ROOT, "ukb_aal")

_FILES = {
    "hold": "data_holdout.npy",
    "2000": "data_2000.npy",
    "4000": "data_4000.npy",
}


def load_aal(variant: str = "hold"):
    """Load a raw AAL variant -> data array (N, T, C)."""
    if variant not in _FILES:
        raise ValueError(f"Unknown UKB AAL variant '{variant}'. Expected one of {list(_FILES)}.")
    return np.load(os.path.join(AAL_DIR, _FILES[variant]))


# --- back-compat alias (used by analysis notebooks) ---
def load_data_hold():
    return load_aal("hold")


def load_pretraining_data(ds_cfg=None):
    """Registry entrypoint. Returns the data array for the configured variant."""
    variant = str(ds_cfg.get("variant", "hold")) if ds_cfg is not None else "hold"
    return load_aal(variant)
