""" UKB ICA dataset loading script (1196 subject holdout data) """

import numpy as np
import pandas as pd
from src.settings import DATA_ROOT

def load_data_hold():

    data = np.load("/data/users2/ppopov1/datasets/ukb_aal/data.npy")

    return data
