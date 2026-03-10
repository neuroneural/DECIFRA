from omegaconf import OmegaConf, DictConfig
import numpy as np
import pandas as pd
import torch 
import os
import time

from src.settings import LOGS_ROOT, DATA_ROOT

from src.datasets.ukb_hold import load_data_hold as load_ukb_pretrain

UKB_DATADICT, demo_df = load_ukb_pretrain(
    file_path=os.path.join(DATA_ROOT, "ukb_ica/ukb_data_hold.npz"),
    demo_path=os.path.join(DATA_ROOT, "ukb_ica/demographics_legend_hold.csv"),
)

UKB_DATA = UKB_DATADICT['data']
print("Available data in ukb datadict:",UKB_DATADICT.keys())
# prepare train and validation sets
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(UKB_DATA, test_size=0.2, random_state=42)

# set model config

from src.models.LSTM_forecaster import default_HPs

if __name__ == "__main__":
    # save the input argument as idx
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int, help="index of the run")
    args = parser.parse_args()
    idx = args.idx

    cfg = {
        "data_info": {
            "feature_size": UKB_DATA.shape[2],
            "n_classes": 2,
            }
        }
    cfg = OmegaConf.create(cfg)

    model_cfg = default_HPs(cfg)

    print(OmegaConf.to_yaml(model_cfg))

    # train
    from src.trainers.BasicPreTrainer import BasicPreTrainer
    from src.models.LSTM_forecaster import LSTM
    import torch
    device = torch.device("cuda")


    model = LSTM(model_cfg)
    model = model.to(device)
    optimizer = model.get_optimizer()
    model.train()

    train_dataloader = model.prepare_pretraining_dataloader(train_data, shuffle=True)
    val_dataloader = model.prepare_pretraining_dataloader(val_data, shuffle=False)



    save_path = f"1_pretrain_{model.__class__.__name__}/{idx:02d}"
    save_path = os.path.join(LOGS_ROOT, save_path)
    
    trainer = BasicPreTrainer(
        cfg=cfg,
        model_cfg=model_cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        epochs=500,
        save_path=save_path,
    )

    train_logs = trainer.run()