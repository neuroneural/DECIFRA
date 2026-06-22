"""
Fine-tuning entrypoint (Hydra): nested-CV classification on top of a pretrained
DECIFRA checkpoint.

Examples
--------
  python scripts/02_finetuning.py model=DECIFRA dataset=fbirn          # all folds
  python scripts/02_finetuning.py model=DECIFRA dataset=fbirn fold=2   # one fold
  python scripts/02_finetuning.py model=DECIFRA dataset=dummy \
         model.finetune.pretrained.load=false                         # from scratch

The pretrained-weights source lives in the model config's finetune.pretrained
block; build_model() handles checkpoint loading by mode. `fold` selects a single
outer fold (for array parallelism) or runs them all when null. All folds of an
experiment share one run dir; results are aggregated by scanning it.
"""
import os
import glob

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from src.settings import LOGS_ROOT
from src.registry import resolve_finetuning_dataset, build_model, build_model_cfg
from src.splits import nested_cv_splits
from src.trainers.FineTuneTrainer import FineTuneTrainer

METRICS = ["test_accuracy", "test_balanced_accuracy", "test_auc", "test_f1_macro"]


def aggregate(run_root):
    """Build fold_results.csv + summary.txt from every completed cell present."""
    rows = []
    for tl in sorted(glob.glob(os.path.join(run_root, "k*", "r*", "test_log.csv"))):
        parts = tl.split(os.sep)
        r = pd.read_csv(tl).iloc[0].to_dict()
        r.update({"fold": int(parts[-3][1:]), "repeat": int(parts[-2][1:])})
        rows.append(r)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["fold", "repeat"])
    df.to_csv(os.path.join(run_root, "fold_results.csv"), index=False)
    n_folds = df["fold"].nunique()
    with open(os.path.join(run_root, "summary.txt"), "w") as f:
        head = f"# {len(df)} runs across {n_folds} folds"
        print("  " + head); f.write(head + "\n")
        for m in METRICS:
            if m in df:
                line = f"{m}: {df[m].mean():.4f} +/- {df[m].std():.4f}"
                print("  " + line); f.write(line + "\n")


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
    run_root = os.path.join(LOGS_ROOT, save_name)   # folds share one experiment dir
    os.makedirs(run_root, exist_ok=True)

    fold_sel = cfg.get("fold")
    fold_sel = None if fold_sel is None else int(fold_sel)
    print(f"Fine-tuning {model_choice} on {dataset_choice} -> {run_root} "
          f"({'all folds' if fold_sel is None else f'fold {fold_sel}'})")

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

    # Safety check: dataset features must match the pretrained checkpoint.
    model_cfg = build_model_cfg(cfg)
    pre = (cfg.model.get("finetune") or {}).get("pretrained") or {}
    if pre.get("run") and pre.get("load", True):
        pre_cfg = OmegaConf.load(os.path.join(pre.run, "model_config.yaml"))
        if int(pre_cfg.input_size) != int(model_cfg.input_size):
            raise ValueError(
                f"feature_size mismatch: dataset has {model_cfg.input_size} features "
                f"but pretrained model expects {pre_cfg.input_size}.")
        if pre_cfg.get("model_name") != model_cfg.model_name:
            print(f"  WARNING: pretrained class {pre_cfg.get('model_name')} != "
                  f"selected {model_cfg.model_name}; loading compatible weights only.")
        print(f"  pretrained: {pre.run} (checkpoint={pre.get('checkpoint', 'best')})")

    t = cfg.train
    batch_size = int(t.batch_size)

    # --- NESTED CV (optionally restricted to one outer fold) ---
    splits = [s for s in nested_cv_splits(labels, int(t.n_splits), int(t.n_repeats),
                                          val_size=t.get("val_size"))
              if fold_sel is None or s[0] == fold_sel]
    print(f"Running {len(splits)} cell(s)")

    for fold, repeat, tr_idx, val_idx, te_idx in splits:
        run_dir = os.path.join(run_root, f"k{fold:02d}", f"r{repeat:02d}")
        os.makedirs(run_dir, exist_ok=True)
        if cfg.resume and os.path.exists(os.path.join(run_dir, "best.pt")) \
                and os.path.exists(os.path.join(run_dir, "test_log.csv")):
            continue

        model, model_cfg = build_model(cfg)   # constructs + loads weights by mode
        model.to(device)
        if fold == splits[0][0] and repeat == 0 and hasattr(model, "_pretrained_info"):
            i = model._pretrained_info
            print(f"  loaded {i['loaded']} tensors from epoch {i['epoch']} "
                  f"({len(i['missing'])} missing e.g. clf)")

        optimizer = model.get_optimizer()
        make = lambda idx, shuffle: type(model).prepare_dataloader(
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
        row = trainer.run()
        print(f"  k{fold:02d} r{repeat:02d}: "
              + " ".join(f"{m.replace('test_','')}={row.get(m, float('nan')):.3f}"
                         for m in METRICS))

    # Aggregate over every completed cell present (handles fold-parallel runs).
    aggregate(run_root)


if __name__ == "__main__":
    main()
