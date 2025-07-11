This repository contains the implementation of DECIFRA architecture designed for derivation of effective connectivity features from fMRI time series.
# 1. Requirements
```bash
conda create -n dcfr python=3.12
conda activate dcfr
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```


# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with the best hyperparams found in the `tune` mode, or with default hyperparams `default_HPs` is set to `True`

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`
- `dataset`: dataset for the experiments. Datasets' config files can be found at `src/conf/dataset`, and their loading scripts are located at `src/datasets`.


## Optional
- `prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory
- `HP_path`: path to custom hyperparams to load
- `follow_splits`: path to an experiment with train/validation/test splits that you want to replicate in the new experiments.
