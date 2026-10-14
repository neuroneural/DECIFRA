# Datasets

Each dataset loading is handled by a module `src/datasets/<name>.py`, selected by the `name` field of a dataset
config (see [`conf/dataset`](../../conf/dataset/README.md)).

## Expected output data format
- `data`: array [subjects, time, regions] (numpy or torch). Time series are z-scored per subject when batched.
- `labels`: one integer per subject, 0..n_classes-1 (fine-tuning only).

## Loader functions
A module defines one or both. Each receives the dataset config `ds_cfg`, so one module can serve
several configs (e.g. data splits or label types).

| Function | Used by | Returns |
|---|---|---|
| `load_pretraining_data(ds_cfg)` | `scripts/01_pretraining.py` | `data`, or `(data, extra_train_data)` to add subjects to the training split only |
| `load_finetuning_data(ds_cfg)` | `scripts/02_finetuning.py`, notebook 03 | `(data, labels)` |

## Example: `dummy.py`
Random data for smoke tests:
```python
def load_dummy_data(n_samples=1000, seq_len=200, n_channels=50):
    return torch.randn(n_samples, seq_len, n_channels)


def load_pretraining_data(ds_cfg=None):
    return load_dummy_data()


def load_finetuning_data(ds_cfg=None):
    n_classes = int(ds_cfg.get("n_classes", 2)) if ds_cfg is not None else 2
    data = load_dummy_data().numpy()
    labels = np.random.default_rng(0).integers(0, n_classes, size=data.shape[0])
    return data, labels
```
For real data, replace `load_dummy_data` with a reader of your files; `fbirn.py` reads its split
and label type from `ds_cfg`.

## Notes
- Pretraining uses one fixed 80/20 train/validation split of `data` (`src/splits.py`).
- The released weights expect the 53 Neuromark 1.0 ICA regions in template order; other
  parcellations need their own pretraining.
- Notebook 03 labels regions and domains from `assets/data/ICN_coordinates_refined.csv` when the
  data have these 53 regions, and by index otherwise.
