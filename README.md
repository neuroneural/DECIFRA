# DECIFRA

This repo contains implementation of DECIFRA model published in IEEE TMI (https://ieeexplore.ieee.org/document/11701636).
(https://doi.org/10.1109/TMI.2026.3735891).

We publish the model weights pretrained on the UKB dataset preprocessed with Neuromark 1.0 ICA pipeline that defines 53 regions.
To test the model on a different data, you need to run pretraining experiments first.
We provide the example scripts necessary for pretraining on the new data. 
To add your dataset to this repo, please check the READMEs in `conf/dataset` and `src/datasets`.
#todo; add these readmes to these folders. i think to keep it simple let's exemplify the instructions using dummy dataset.

## Installation
```bash
git clone https://github.com/neuroneural/DECIFRA.git && cd DECIFRA
pip install -r requirements.txt   # TODO: get the version numbers from conda env pile and insert them there
```

To see if it works, run this from the repo root:
```bash
PYTHONPATH=. python scripts/01_pretraining.py model=DECIFRA dataset=dummy train.epochs=2
```

## Pretraining
Example: #todo add reference to the pretrain config for more details on params
```bash
PYTHONPATH=. python scripts/01_pretraining.py model=DECIFRA_MS/default dataset=ukb_ica idx=0
```
- Models: `DECIFRA`, `DECIFRA_MS/default`; ablations in `conf/model/DECIFRA_MS/`.
- Logs: `assets/logs/1_pretrain-<dataset>-<model>/<idx>/` (checkpoints, `best_epoch.txt`, `train_logs.csv`).
- SLURM: `scripts/slurm/01_pretrain_*.sh`.

#todo; explain what to change to use another dataset for pretraining, and how idx works to run multiple experiments (and what happens if we don't set it)

## Fine-tuning
Example:  #todo add reference to the finetune config for more details on params
```bash
PYTHONPATH=. python scripts/02_finetuning.py model=DECIFRA dataset=fbirn +dataset@test_sets.holdout=fbirn_holdout
```
- Pretrained weights: `finetune.pretrained` in `conf/model/<model>.yaml` (`run`, `checkpoint`; `load: false` trains from scratch). TODO: released weights. assets/weights let's put them here, copy them from logs and name like model_dataset_idx.pth
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

## Using your own data 
#let's move this info to the READMEs I mention above, no need for it here
- Arrays: `data` [subjects, time, regions], `labels` [subjects] (integers, fine-tuning only).
- `src/datasets/<name>.py` with `load_pretraining_data(ds_cfg) -> data` and/or
  `load_finetuning_data(ds_cfg) -> (data, labels)` (see `dummy.py`, `fbirn.py`).
- `conf/dataset/<name>.yaml` with `name: <name>`, `n_classes` (fine-tuning) and any keys your loader reads.
- Region names and domains for plots: `assets/data/ICN_coordinates_refined.csv` (Neuromark 53); other parcellations are shown by index.


