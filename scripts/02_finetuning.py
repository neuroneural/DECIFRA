"""
Fine-tuning entrypoint (Hydra): nested-CV classification on top of a pretrained
DECIFRA checkpoint.

Examples
--------
  python scripts/02_finetuning.py model=DECIFRA_MS/default dataset=fbirn \
         pretrained.run=assets/logs/1_pretrain-ukb-DECIFRA_MS-default/00 idx=0
  python scripts/02_finetuning.py model=DECIFRA/default dataset=dummy \
         pretrained.load=false idx=0          # classifier from scratch (baseline)

For each outer fold (held-out test) and inner repeat (train/val resample) a fresh
model is built, pretrained weights are loaded (strict=False, clf head skipped),
and a single train/val/test cycle runs. Results are aggregated to the run root.
"""
import os

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from src.settings import LOGS_ROOT
from src.registry import (
    resolve_finetuning_dataset, resolve_model, build_model_cfg, load_pretrained_state,
)
from src.splits import nested_cv_splits
from src.trainers.FineTuneTrainer import FineTuneTrainer

METRICS = ["test_accuracy", "test_balanced_accuracy", "test_auc", "test_f1_macro"]


@hydra.main(version_base=None, config_path="../conf", config_name="finetune")
def main(cfg):
    choices = HydraConfig.get().runtime.choices
    model_choice, dataset_choice = choices["model"], choices["dataset"]

    variant = cfg.model.get("variant", None)
    tag = model_choice.replace("/", "-")
    if variant is not None and str(variant) != "default":
        tag += f"-{variant}"
    save_name = f"2_finetune-{dataset_choice}-{tag}"
    if cfg.get("postfix"):
        save_name += f"-{cfg.postfix}"
    run_root = os.path.join(LOGS_ROOT, save_name, f"{int(cfg.idx):02d}")
    os.makedirs(run_root, exist_ok=True)
    print(f"Fine-tuning config {model_choice} on {dataset_choice} -> {run_root}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- DATA + LABELS ---
    data, labels = resolve_finetuning_dataset(cfg)
    data = np.asarray(data)
    labels = np.asarray(labels)
    n_classes = int(cfg.dataset.get("n_classes", len(np.unique(labels))))
    print(f"Data {data.shape}, labels {labels.shape}, classes {n_classes}, "
          f"balance {np.bincount(labels).tolist()}")

    with open_dict(cfg):
        cfg.data_info = {"feature_size": int(data.shape[2]), "n_classes": n_classes}

    # --- MODEL TEMPLATE (built fresh per run; here just for the safety check) ---
    ModelClass = resolve_model(cfg)
    model_cfg = build_model_cfg(cfg)

    # Safety check against the pretrained run's saved architecture.
    pre = cfg.pretrained
    if pre.get("run") and pre.get("load", True):
        pre_cfg = OmegaConf.load(os.path.join(pre.run, "model_config.yaml"))
        if int(pre_cfg.input_size) != int(model_cfg.input_size):
            raise ValueError(
                f"feature_size mismatch: dataset has {model_cfg.input_size} features "
                f"but pretrained model expects {pre_cfg.input_size}.")
        if pre_cfg.get("model_name") != model_cfg.model_name:
            print(f"  WARNING: pretrained class {pre_cfg.get('model_name')} != "
                  f"selected {model_cfg.model_name}; loading compatible weights only.")

    t = cfg.train
    batch_size = int(t.batch_size)

    # --- NESTED CV ---
    results = []
    splits = list(nested_cv_splits(labels, int(t.n_splits), int(t.n_repeats),
                                   val_size=t.get("val_size")))
    print(f"Running {len(splits)} runs ({t.n_splits} folds x {t.n_repeats} repeats)")

    for fold, repeat, tr_idx, val_idx, te_idx in splits:
        run_dir = os.path.join(run_root, f"k{fold:02d}", f"r{repeat:02d}")
        os.makedirs(run_dir, exist_ok=True)

        if cfg.resume and os.path.exists(os.path.join(run_dir, "best.pt")) \
                and os.path.exists(os.path.join(run_dir, "test_log.csv")):
            row = pd.read_csv(os.path.join(run_dir, "test_log.csv")).iloc[0].to_dict()
            row.update({"fold": fold, "repeat": repeat})
            results.append(row)
            continue

        model = ModelClass(model_cfg).to(device)
        if pre.get("run") and pre.get("load", True):
            info = load_pretrained_state(model, pre.run, pre.checkpoint, list(pre.drop_keys))
            if fold == 0 and repeat == 0:
                print(f"  loaded {info['loaded']} tensors from epoch {info['epoch']} "
                      f"({len(info['missing'])} missing e.g. clf)")

        optimizer = model.get_optimizer()
        make = lambda idx, shuffle: ModelClass.prepare_dataloader(
            data[idx], labels[idx], shuffle=shuffle, batch_size=batch_size, zscore=t.zscore)

        trainer = FineTuneTrainer(
            cfg=cfg, model_cfg=model_cfg, model=model, optimizer=optimizer,
            train_loader=make(tr_idx, True),
            val_loader=make(val_idx, False),
            test_loader=make(te_idx, False),
            epochs=int(t.epochs), patience=t.get("patience"),
            device=str(device), save_path=run_dir, resume=cfg.resume,
            preserve_checkpoints=bool(t.preserve_checkpoints),
        )
        test_row = trainer.run()
        test_row.update({"fold": fold, "repeat": repeat})
        results.append(test_row)
        print(f"  k{fold:02d} r{repeat:02d}: "
              + " ".join(f"{m.replace('test_','')}={test_row.get(m, float('nan')):.3f}"
                         for m in METRICS))

    # --- AGGREGATE ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_root, "fold_results.csv"), index=False)
    summary = {m: (df[m].mean(), df[m].std()) for m in METRICS if m in df}
    with open(os.path.join(run_root, "summary.txt"), "w") as f:
        for m, (mean, std) in summary.items():
            line = f"{m}: {mean:.4f} +/- {std:.4f}"
            print("  " + line)
            f.write(line + "\n")
    return df


if __name__ == "__main__":
    main()
