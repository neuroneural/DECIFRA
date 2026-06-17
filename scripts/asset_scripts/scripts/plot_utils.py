import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _nullify_diag_if_needed(data):
    """
    If any element on the diagonal of the matrix (or the first matrix in a batch)
    is close to 1.0, set all diagonals to 0.0 to prevent colorbar distortion.
    """
    res = data.copy()
    if res.ndim == 2:
        if np.any(np.isclose(np.diag(res), 1.0, atol=0.1)):
            np.fill_diagonal(res, 0.0)
    elif res.ndim >= 3:
        first_mat = res.reshape(-1, res.shape[-2], res.shape[-1])[0]
        if np.any(np.isclose(np.diag(first_mat), 1.0, atol=0.1)):
            mask = np.eye(res.shape[-1], dtype=bool)
            res[..., mask] = 0.0
    return res

def get_domain_boundaries(csv_path="/Users/ppopov1/DECIFRA/assets/data/ICN_coordinates_refined.csv"):
    df = pd.read_csv(csv_path)
    domains = df['Domain'].values
    
    transitions = []
    labels = []
    
    current_domain = domains[0]
    labels.append((current_domain, 0))
    
    for i in range(1, len(domains)):
        if domains[i] != current_domain:
            transitions.append(i - 0.5)
            current_domain = domains[i]
            labels.append((current_domain, i))
            
    return transitions, labels

def plot_domain_matrix(matrix, ax, title, transitions, vmin=None, vmax=None, cmap=None):
    matrix_plot = _nullify_diag_if_needed(matrix)
    
    if cmap is None:
        cmap = "inferno" if np.min(matrix_plot) >= 0 else "seismic"
        
    if vmin is None or vmax is None:
        if cmap == "inferno":
            vmin = 0
            vmax = np.max(matrix_plot)
            if vmax == 0: vmax = 1e-6
        else:
            abs_max = np.max(np.abs(matrix_plot))
            if abs_max == 0:
                abs_max = 1e-6
            vmin, vmax = -abs_max, abs_max
            
    im = ax.imshow(matrix_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    
    line_color = 'silver' if cmap == 'inferno' else 'black'
    for t in transitions:
        ax.axhline(t, color=line_color, linewidth=1)
        ax.axvline(t, color=line_color, linewidth=1)
        
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_comparison_figure(mat_true, mat_sal, mat_ig, title_prefix="", colorbar=True):
    transitions, labels = get_domain_boundaries()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = plot_domain_matrix(mat_true, axes[0], f"{title_prefix} True Transfer", transitions)
    if colorbar:
        plt.colorbar(im1, ax=axes[0])
    
    im2 = plot_domain_matrix(mat_sal, axes[1], f"{title_prefix} Saliency Map", transitions)
    if colorbar:
        plt.colorbar(im2, ax=axes[1])
    
    im3 = plot_domain_matrix(mat_ig, axes[2], f"{title_prefix} Integrated Gradients", transitions)
    if colorbar:
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def plot_first_n_matrices(tensor, title, n_time=5, colorbar=False, individual_limits=False):
    # tensor shape: [T, C, C]
    n_time = min(n_time, len(tensor))
    transitions, labels = get_domain_boundaries()
    
    tensor_plot = _nullify_diag_if_needed(tensor)
    
    fig, axes = plt.subplots(1, n_time, figsize=(4*n_time, 4))
    if n_time == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    
    if not individual_limits:
        if np.min(tensor_plot[:n_time]) >= 0:
            cmap = "inferno"
            vmin = 0
            vmax = np.max(tensor_plot[:n_time])
            if vmax == 0: vmax = 1e-6
        else:
            cmap = "seismic"
            abs_max = np.max(np.abs(tensor_plot[:n_time]))
            if abs_max == 0: abs_max = 1e-6
            vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax, cmap = None, None, None
    
    for t in range(n_time):
        im = plot_domain_matrix(tensor_plot[t], axes[t], f"Time {t}", transitions, vmin=vmin, vmax=vmax, cmap=cmap)
        if colorbar and individual_limits:
            fig.colorbar(im, ax=axes[t], fraction=0.046, pad=0.04)
    
    if colorbar and not individual_limits:
        # Add a single colorbar to the right
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
