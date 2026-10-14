# Dataset configs

`dataset=<config>` selects `conf/dataset/<config>.yaml`. Its fields are passed to the dataset's
loader (see [`src/datasets`](../../src/datasets/README.md)), and log folders are named after the config.

| Field | Meaning |
|---|---|
| `name` | module `src/datasets/<name>.py` that loads the data |
| `n_classes` | number of classes (fine-tuning) |
| any other | read by the loader, e.g. `split` and `label` in `fbirn.yaml` |

Several configs can share a module: `ukb.yaml`, `ukb_half.yaml` and `ukb_plus_1000.yaml` all use
`name: ukb_ica` with different `variant`s.

## Example: `dummy.yaml`
```yaml
name: dummy
n_classes: 2
```
Pretrain on it, then fine-tune from that run:
```bash
PYTHONPATH=. python scripts/01_pretraining.py model=DECIFRA dataset=dummy train.epochs=2
PYTHONPATH=. python scripts/02_finetuning.py model=DECIFRA dataset=dummy train.epochs=2 \
    train.n_splits=2 train.n_repeats=1 model.finetune.pretrained.run=assets/logs/1_pretrain-dummy-DECIFRA/00
```
