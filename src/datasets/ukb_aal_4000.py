""" UKB AAL dataset loading script (4000 subjects data) """

import os
import numpy as np
from src.settings import DATA_ROOT

def load_data_4000():
    data = np.load(os.path.join(DATA_ROOT, "ukb_aal", "data_4000.npy"))
    return data
