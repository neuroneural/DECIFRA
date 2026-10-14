import numpy as np


def zscore_np(x, axis=0, eps=1e-8):
    """
    numpy-only z-score

    Parameters
    ----------
    x : array_like
        data to z-score
    axis : int or tuple of int
        Axis/axes along which to compute mean/std. (e.g., axis=1 for (B,T,D) time axis)

    Returns
    -------
    xz : ndarray
        Z-scored array (same shape as x).
    """
    x = np.asarray(x)
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def corrcoef_batch(x, eps=1e-8):
    """
    Batched Pearson correlation over features for time-series data.

    Parameters
    ----------
    X : array_like, shape (B, T, D)
        Data to correlate, arranged as Batch x Time x (D)Features.

    Returns
    -------
    C : ndarray, shape (B, D, D)
        Pearson correlation matrices per batch item.
    """
    x = np.asarray(x)
    B, T, D = x.shape

    # z-score along time
    Z = zscore_np(x, axis=1, eps=eps)   # (B, T, D)

    # Correlation = (Z^T @ Z) / T for each batch element
    C = np.einsum('btd,bte->bde', Z, Z) / T             # (B, D, D)

    # Symmetrize & fix diagonals (nice-to-have for numeric drift)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    batch_indices = np.arange(B)[:, None]
    diag_indices = np.arange(D)
    C[batch_indices, diag_indices, diag_indices] = 1.0

    return C