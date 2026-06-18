"""
UKB ICA dataset (53 ICA components), consolidated across all variants.

Variants (selected via the dataset config's `variant` field):
  - hold       : 1196-subject holdout       (ukb_data_hold.npz)
  - exp        : experiment split            (ukb_data_exp.npz)
  - 2205       : holdout downsampled in time (every 3rd timepoint)
  - half       : holdout, first half of the time series
  - plus_1000  : holdout + first 1000 exp subjects appended to TRAIN only
                 (returns (data, extra_train_data))

`load_pretraining_data(ds_cfg)` is the registry entrypoint. `load_ica(variant)`
returns the full record dict (data, sexes, ages, age_bins) + demographics frame
for downstream use (e.g. fine-tuning labels). `load_data_hold`/`load_data_exp`
are kept as back-compat aliases.
"""
import os

import numpy as np
import pandas as pd

from src.settings import DATA_ROOT

ICA_DIR = os.path.join(DATA_ROOT, "ukb_ica")

# variant feeding off a raw npz -> (filename, demographics csv)
_NPZ = {
    "hold": ("ukb_data_hold.npz", "demographics_legend_hold.csv"),
    "exp": ("ukb_data_exp.npz", "demographics_legend_exp.csv"),
}


def load_ica(variant: str = "hold"):
    """Load a raw ICA npz variant -> (record dict, demographics DataFrame)."""
    if variant not in _NPZ:
        raise ValueError(f"Unknown raw ICA variant '{variant}'. Expected one of {list(_NPZ)}.")
    fname, demo_name = _NPZ[variant]
    npz = np.load(os.path.join(ICA_DIR, fname), allow_pickle=True)
    record = {
        "data": npz["data"],        # (N, T, 53)
        "sexes": npz["sexes"],
        "ages": npz["ages"],
        "age_bins": npz["age_bins"],
    }
    demo_df = pd.read_csv(os.path.join(ICA_DIR, demo_name))
    return record, demo_df


# --- back-compat aliases (used by analysis notebooks) ---
def load_data_hold():
    return load_ica("hold")


def load_data_exp():
    return load_ica("exp")


def load_pretraining_data(ds_cfg=None):
    """Registry entrypoint. Returns `data` or `(data, extra_train_data)`."""
    variant = str(ds_cfg.get("variant", "hold")) if ds_cfg is not None else "hold"

    if variant in ("hold", "exp"):
        return load_ica(variant)[0]["data"]

    if variant == "2205":
        return load_ica("hold")[0]["data"][:, ::3, :]

    if variant == "half":
        data = load_ica("hold")[0]["data"]
        return data[:, : data.shape[1] // 2, :]

    if variant == "plus_1000":
        hold = load_ica("hold")[0]["data"]
        extra_train = load_ica("exp")[0]["data"][:1000]
        return hold, extra_train

    raise ValueError(
        f"Unknown UKB ICA variant '{variant}'. "
        f"Expected: hold, exp, 2205, half, plus_1000."
    )
