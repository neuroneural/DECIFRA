""" UKB ICA dataset loading script (1196 subject holdout data) """

import numpy as np
import pandas as pd
from src.settings import DATA_ROOT


def load_data_exp(
    file_path: str = f"{DATA_ROOT}/ukb_ica/ukb_data_exp.npz",
    demo_path: str = f"{DATA_ROOT}/ukb_ica/demographics_legend_exp.csv"
    ):
    """
    Loads UKB ICA data (experiment) saved in npz archive.

    Args:
        file_path (str): The path to the .npz file.
        demo_path (str): The path to the demographics legend CSV file.

    Returns:
        A dictionary containing the loaded arrays:
            - data: The main data array of shape (34656, 490, 53) (samples, time points, features).
            - sexes: Sex labels.    
            - ages: Age values.
            - age_bins: Age bin labels.
        A pandas DataFrame containing the demographics legend.
    """
    npz = np.load(file_path, allow_pickle=True)
    data = npz["data"]
    sexes = npz["sexes"]
    ages = npz["ages"]
    age_bins = npz["age_bins"]

    demo_df = pd.read_csv(demo_path)

    return {
        "data": data,
        "sexes": sexes,
        "ages": ages,
        "age_bins": age_bins
    }, demo_df

def load_data_hold_2205():
    data_dict, demo_df = load_data_exp()
    data = data_dict["data"]
    data = data[:, ::3, :]
    return data

if __name__ == "__main__":
    # Example of how to use the function to load the data
    data_dict, demo_df = load_data_exp()

    # Print the keys of the loaded data to verify
    print("Arrays loaded from the file:", data_dict.keys())
    print("Data shape:", data_dict["data"].shape)
    print("Others shape:", data_dict["sexes"].shape, data_dict["ages"].shape, data_dict["age_bins"].shape)
    print("Demographics legend DataFrame:")
    print(demo_df)
