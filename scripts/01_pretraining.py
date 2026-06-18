"""
Pretraining entrypoint (Hydra).

Examples
--------
  python scripts/01_pretraining.py model=DECIFRA_MS/default dataset=ukb idx=0
  python scripts/01_pretraining.py model=VAR/lag_5_full dataset=ukb idx=3 train.epochs=300
  python scripts/01_pretraining.py model=DECIFRA_sens/sp_heavy dataset=ukb idx=0 resume=true

Log dir: assets/logs/1_pretrain-<dataset>-<model_choice>/<idx>/  (model_choice with
'/' replaced by '-', e.g. DECIFRA_MS-default), plus an optional `postfix`.
"""
import os

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import train_test_split

from src.settings import LOGS_ROOT
from src.registry import resolve_dataset, resolve_model, build_model_cfg


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg):
    choices = HydraConfig.get().runtime.choices
    model_choice = choices["model"]      # e.g. "DECIFRA_MS/default"
    dataset_choice = choices["dataset"]  # e.g. "ukb"

    tag = model_choice.replace("/", "-")
    save_name = f"1_pretrain-{dataset_choice}-{tag}"
    if cfg.get("postfix"):
        save_name += f"-{cfg.postfix}"
    save_path = os.path.join(LOGS_ROOT, save_name, f"{int(cfg.idx):02d}")

    print(f"Running {cfg.model.model_name} (module src.models.{cfg.model.get('module') or cfg.model.model_name}) "
          f"on {dataset_choice} (config {model_choice}, idx {cfg.idx})")
    print(f"Saving to: {save_path}")

    epochs = int(cfg.train.epochs)
    batch_size = int(cfg.train.batch_size)
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

    # --- DATA ---
    data, more_tr_data = resolve_dataset(cfg)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    if more_tr_data is not None:
        train_data = np.concatenate([train_data, more_tr_data], axis=0)
    print(f"Train data shape: {train_data.shape}, Val data shape: {val_data.shape}")

    # runtime sizes consumed by build_model_cfg (pretraining omits n_classes)
    with open_dict(cfg):
        cfg.data_info = {"feature_size": int(data.shape[2])}

    # --- MODEL ---
    ModelClass = resolve_model(cfg)
    model_cfg = build_model_cfg(cfg)
    model = ModelClass(model_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {n_params}")
    with open_dict(cfg):
        cfg.n_params = n_params

    # --- TRAIN ---
    from src.trainers.BasicPreTrainer import BasicPreTrainer, find_optimal_batch_size
    from src.trainers.StagePreTrainer import StagePreTrainer

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    if batch_size <= 0:
        batch_size = find_optimal_batch_size(model, train_data, device)
        with open_dict(cfg):
            cfg.train.batch_size = batch_size
        print(f"\nFinal optimal batch size utilized: {batch_size}")

    optimizer = model.get_optimizer()
    model.train()

    train_loader = model.prepare_pretraining_dataloader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = model.prepare_pretraining_dataloader(val_data, shuffle=False, batch_size=batch_size)

    n_stages = model_cfg.get("n_training_stages", 1)
    if n_stages > 1:
        ratios = model_cfg.get("eps_stage_ratios", [1.0 / n_stages] * n_stages)
        stage_epochs_list = [max(1, int(r * epochs)) for r in ratios]
        stage_epochs_list[-1] = epochs - sum(stage_epochs_list[:-1])
        print(f"Using StagePreTrainer with {n_stages} stages: {stage_epochs_list}")
        trainer = StagePreTrainer(
            cfg=cfg, model_cfg=model_cfg, model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            stage_epochs_list=stage_epochs_list, save_path=save_path, resume=cfg.resume,
        )
    else:
        trainer = BasicPreTrainer(
            cfg=cfg, model_cfg=model_cfg, model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            epochs=epochs, save_path=save_path, resume=cfg.resume,
        )

    trainer.run()


if __name__ == "__main__":
    main()
