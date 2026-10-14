# DECIFRA

This repo contains implementation of DECIFRA model published in IEEE TMI (https://ieeexplore.ieee.org/document/11701636).
(https://doi.org/10.1109/TMI.2026.3735891).

We publish the model weights pretrained on the UKB dataset preprocessed with Neuromark 1.0 ICA pipeline that defines 53 regions.
To test the model on a different data, you need to run pretraining experiments first.
We provide the example scripts necessary for pretraining on the new data. 
To add your dataset to this repo, please check the READMEs in [`conf/dataset`](conf/dataset/README.md) and [`src/datasets`](src/datasets/README.md).

## Installation
```bash
git clone https://github.com/neuroneural/DECIFRA.git && cd DECIFRA
pip install -r requirements.txt   # tested with Python 3.12
```

To see if it works, run this from the repo root:
```bash
PYTHONPATH=. python scripts/01_pretraining.py model=DECIFRA dataset=dummy train.epochs=2
```

## Pretraining
Example:
```bash
PYTHONPATH=. python scripts/01_pretraining.py model=DECIFRA_MS/default dataset=ukb idx=0
```
Parameters and defaults are stored in `conf/pretrain.yaml` and `conf/model/<model>.yaml`; 
override any of them on the command line, e.g. `train.epochs=100`)

- Models: `DECIFRA`, `DECIFRA_MS/default`; ablations in `conf/model/DECIFRA_MS/`.
- Logs: `assets/logs/1_pretrain-<dataset>-<model>/<idx>/` (checkpoints, `best_epoch.txt`, `train_logs.csv`).
- SLURM: `scripts/slurm/01_pretrain_*.sh`.
- Other data: `dataset=<config>` selects `conf/dataset/<config>.yaml` (see [`conf/dataset`](conf/dataset/README.md)).
- `idx` (default 0) names the run folder. Sweep through multiple values to run multiple experiments in parallel.

Parameters and defaults are stored in `conf/finetune.yaml` and the `finetune` block of `conf/model/<model>.yaml`).

## Fine-tuning
Example:
```bash
PYTHONPATH=. python scripts/02_finetuning.py model=DECIFRA dataset=fbirn +dataset@test_sets.holdout=fbirn_holdout
```
- Pretrained weights: `finetune.pretrained.run` in `conf/model/<model>.yaml`, by default the released UKB weights `assets/weights/<model>_<dataset>_<idx>.pth`. It also takes a pretraining run folder (`checkpoint: best` or an epoch); `load: false` trains from scratch.
- CV setup (`train.n_splits` × `train.n_repeats`); each cell's best-by-validation model is scored on its test fold and on every `test_sets` entry.
- Logs: `assets/logs/2_finetune-<dataset>-<model>/k*/r*/`, plus `fold_results.csv` and `summary.txt`; `resume=true` skips finished cells.
- SLURM: `scripts/slurm/02_finetune_*.sh`.

## Analysis
Use the notebooks below to analyze the results of pretraining or finetuning.

| Notebook (`scripts/notebooks/`) | Content |
|---|---|
| `01_pretraining_analysis` | best pretraining run, losses, matrices, attractor analysis |
| `02_finetuning_analysis` | classification scores, learning curves |
| `03_matrix_analysis` | functional patterns, class-specific states, second-order dynamics, connectograms |

## Repository layout
```
conf/      Hydra configs: pretrain.yaml, finetune.yaml, model/, dataset/
src/       models/, datasets/, trainers/, registry.py, settings.py
scripts/   01_pretraining.py, 02_finetuning.py, notebooks/, slurm/
assets/    data/, logs/, analysis/
```

## Citation
```bibtex
@article{decifra_early_access,
  author={Popov, Pavel and Morasso, Cristian and Dolci, Giorgio and Galazzo, Ilaria Boscolo and Pearlson, Godfrey and Danks, David and Calhoun, Vince and Menegaz, Gloria and Plis, Sergey},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Message in a Bottleneck: Interpretable Deep Learning for Dynamic Effective Brain Connectivity}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2026.3735891}
}
```

## License
MIT, see [LICENSE](LICENSE).
