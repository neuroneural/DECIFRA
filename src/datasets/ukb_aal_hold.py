""" UKB AAL dataset loading script (holdout data) """

import os
import numpy as np
from ..settings import DATA_ROOT

def load_data_hold():
    data = np.load(os.path.join(DATA_ROOT, "ukb_aal", "data_holdout.npy"))
    return data
