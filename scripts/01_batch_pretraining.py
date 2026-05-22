from omegaconf import OmegaConf, DictConfig
import numpy as np
import pandas as pd
import torch 
import os
import time

from src.settings import LOGS_ROOT, DATA_ROOT

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run configuration for my model")

    parser.add_argument("--model", type=str, required=True, help="Name of the model architecture")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--idx", type=int, required=True, help="index of the run")
    parser.add_argument("--postfix", type=str, help="Postfix for save path", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size for training (0 for auto-detection)", default=64)
    parser.add_argument("--epochs", type=int, help="Number of epochs for training", default=500)
    parser.add_argument("--hp_config", type=str, help="Optional path to a .yaml hyperparameter config file", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last completed epoch if directory exists")

    args = parser.parse_args()
    model_name, ds_name = args.model, args.dataset
    print(f"Running {args.model} on {args.dataset} (Index: {args.idx})")

    idx, postfix = args.idx, args.postfix
    SAVE_PATH = f"1_pretrain-{ds_name}-{model_name}" if postfix is None else f"1_pretrain-{ds_name}-{model_name}-{postfix}"
    SAVE_PATH = os.path.join(LOGS_ROOT, SAVE_PATH, f"{idx:02d}")
    print(f"Saving to: {SAVE_PATH}")

    batch_size, epochs = args.batch_size, args.epochs
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

    # --- SET DATASET ---
    if ds_name == "ukb":
        from src.datasets.ukb_hold import load_data_hold as load_ukb_pretrain
        UKB_DATADICT, demo_df = load_ukb_pretrain(
            file_path=os.path.join(DATA_ROOT, "ukb_ica/ukb_data_hold.npz"),
            demo_path=os.path.join(DATA_ROOT, "ukb_ica/demographics_legend_hold.csv"),
        )
        data = UKB_DATADICT['data']
    elif ds_name == "ukb_exp":
        from src.datasets.ukb_exp import load_data_exp as load_ukb_exp_pretrain
        UKB_DATADICT, demo_df = load_ukb_exp_pretrain(
            file_path=os.path.join(DATA_ROOT, "ukb_ica/ukb_data_exp.npz"),
            demo_path=os.path.join(DATA_ROOT, "ukb_ica/demographics_legend_exp.csv"),
        )
        data = UKB_DATADICT['data']
    elif ds_name == "ukb_aal_hold":
        from src.datasets.ukb_aal_hold import load_data_hold as load_ukb_aal_hold_pretrain
        data = load_ukb_aal_hold_pretrain()
    elif ds_name == "ukb_aal_2000":
        from src.datasets.ukb_aal_2000 import load_data_2000 as load_ukb_aal_2000_pretrain
        data = load_ukb_aal_2000_pretrain()
    elif ds_name == "ukb_aal_4000":
        from src.datasets.ukb_aal_4000 import load_data_4000 as load_ukb_aal_4000_pretrain
        data = load_ukb_aal_4000_pretrain()
    elif ds_name == "ukb_2205":
        from src.datasets.ukb_hold import load_data_hold_2205 as load_ukb_2205_pretrain
        data = load_ukb_2205_pretrain()
    elif ds_name == "ukb_half":
        from src.datasets.ukb_hold import load_data_hold_half
        _, data = load_data_hold_half()
    elif ds_name == "ukb_plus_1000":
        from src.datasets.ukb_hold import load_data_hold_plus_1000
        data, more_tr_data = load_data_hold_plus_1000()
    elif ds_name == "dummy":
        from src.datasets.dummy import load_dummy_data
        data = load_dummy_data()

    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    # prepare train and validation sets
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    if ds_name == "ukb_plus_1000":
        train_data = np.concatenate([train_data, more_tr_data], axis=0)
    print(f"Train data shape: {train_data.shape}, Val data shape: {val_data.shape}")

    # --- SET MODEL ---
    hp_config_path = args.hp_config

    if model_name == "DECIFRA":
        from src.models.DECIFRA import DECIFRA as ModelClass, default_HPs, custom_HPs
    elif model_name == "GRU_forecaster":
        from src.models.GRU_forecaster import GRU_forecaster as ModelClass, default_HPs, custom_HPs
    elif model_name == "LSTM_forecaster":
        from src.models.LSTM_forecaster import LSTM_forecaster as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_noGate":
        from src.models.DECIFRA import DECIFRA_noGate as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_noGate_IMix_Res":
        from src.models.DECIFRA import DECIFRA_noGate_IMix_Res as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_IMix":
        from src.models.DECIFRA import DECIFRA_IMix as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_IMix_Res":
        from src.models.DECIFRA import DECIFRA_IMix_Res as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_Gated_IMix_Res":
        from src.models.DECIFRA import DECIFRA_Gated_IMix_Res as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_MS":
        from src.models.DECIFRA_MS import DECIFRA_MS as ModelClass, default_HPs, custom_HPs
    elif model_name == "DECIFRA_rand":
        from src.models.DECIFRA_rand import DECIFRA_rand as ModelClass, default_HPs, custom_HPs
    elif model_name == "meanGRU":
        from src.models.meanGRU import meanGRU as ModelClass, default_HPs, custom_HPs
    elif model_name == "meanLSTM":
        from src.models.meanLSTM import meanLSTM as ModelClass, default_HPs, custom_HPs
    elif model_name == "VAR":
        from src.models.VAR import VAR as ModelClass, default_HPs, custom_HPs
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg = {
        "model": model_name,
        "dataset": ds_name,
        "batch_size": batch_size,
        "idx": idx,
        "epochs": epochs,
        "data_info": {
            "feature_size": data.shape[2],
            "n_classes": 2,
        }
    }
    cfg = OmegaConf.create(cfg)
    
    # Init hyperparameters
    if hp_config_path:
        print(f"Loading custom HPs from: {hp_config_path}")
        model_cfg = custom_HPs(cfg, hp_config_path)
    else:
        model_cfg = default_HPs(cfg)
        
    model = ModelClass(model_cfg)
    # print(OmegaConf.to_yaml(model_cfg))
    # add model parameter count to config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {n_params}")
    cfg.n_params = n_params

    # --- TRAIN ---
    from src.trainers.BasicPreTrainer import BasicPreTrainer, find_optimal_batch_size
    from src.trainers.StagePreTrainer import StagePreTrainer
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if batch_size <= 0:
        batch_size = find_optimal_batch_size(model, train_data, device)
        cfg.batch_size = batch_size
        print(f"\nFinal optimal batch size utilized: {batch_size}")

    optimizer = model.get_optimizer()
    model.train()

    train_dataloader = model.prepare_pretraining_dataloader(train_data, shuffle=True, batch_size=batch_size)
    val_dataloader = model.prepare_pretraining_dataloader(val_data, shuffle=False, batch_size=batch_size)
    
    # Select Trainer
    n_stages = model_cfg.get("n_training_stages", 1)
    if n_stages > 1:
        # Calculate stage epochs
        ratios = model_cfg.get("eps_stage_ratios", [1.0/n_stages]*n_stages)
        stage_epochs_list = [max(1, int(r * epochs)) for r in ratios]
        # Adjust last stage to match total epochs exactly
        stage_epochs_list[-1] = epochs - sum(stage_epochs_list[:-1])
        
        print(f"Using StagePreTrainer with {n_stages} stages: {stage_epochs_list}")
        trainer = StagePreTrainer(
            cfg=cfg,
            model_cfg=model_cfg,
            model=model,
            optimizer=optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            stage_epochs_list=stage_epochs_list,
            save_path=SAVE_PATH,
            resume=args.resume,
        )
    else:
        trainer = BasicPreTrainer(
            cfg=cfg,
            model_cfg=model_cfg,
            model=model,
            optimizer=optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=epochs,
            save_path=SAVE_PATH,
            resume=args.resume,
        )

    train_logs = trainer.run()