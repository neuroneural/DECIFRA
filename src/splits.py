"""
Nested cross-validation splits for fine-tuning (port of DECIFRA_old's scheme).

Outer: StratifiedKFold(n_splits) -> (train+val pool, test).
Inner: StratifiedShuffleSplit(n_repeats) on the pool -> (train, val), a different
       split per repeat. Test is the held-out outer fold.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def nested_cv_splits(labels, n_splits, n_repeats, val_size=None, seed=42):
    """
    Yield ``(fold, repeat, train_idx, val_idx, test_idx)`` over the full nested CV.

    Parameters
    ----------
    labels : array-like
        Class labels, used for stratification.
    n_splits : int
        Number of outer folds (each fold's held-out part is the test set).
    n_repeats : int
        Number of train/val resamples per fold.
    val_size : float | None
        Fraction of the train+val pool used for validation. Defaults to
        ``1 / n_splits`` (matches the old codebase's val proportion).
    seed : int
        RNG seed for both the outer and inner splitters.
    """
    labels = np.asarray(labels)
    if val_size is None:
        val_size = 1.0 / n_splits

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (pool_idx, test_idx) in enumerate(outer.split(labels, labels)):
        y_pool = labels[pool_idx]
        inner = StratifiedShuffleSplit(
            n_splits=n_repeats, test_size=val_size, random_state=seed
        )
        for repeat, (tr_rel, val_rel) in enumerate(inner.split(y_pool, y_pool)):
            yield fold, repeat, pool_idx[tr_rel], pool_idx[val_rel], test_idx
