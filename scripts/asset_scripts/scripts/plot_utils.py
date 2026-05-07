import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_domain_matrix(matrix, ax, title, transitions, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(matrix))
        if abs_max == 0:
            abs_max = 1e-6
        vmin, vmax = -abs_max, abs_max
        
    im = ax.imshow(matrix, cmap="seismic", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    
    for t in transitions:
        ax.axhline(t, color='black', linewidth=1)
        ax.axvline(t, color='black', linewidth=1)
        
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_comparison_figure(mat_true, mat_sal, mat_ig, title_prefix=""):
    transitions, labels = get_domain_boundaries()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = plot_domain_matrix(mat_true, axes[0], f"{title_prefix} True Transfer", transitions)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = plot_domain_matrix(mat_sal, axes[1], f"{title_prefix} Saliency Map", transitions)
    plt.colorbar(im2, ax=axes[1])
    
    im3 = plot_domain_matrix(mat_ig, axes[2], f"{title_prefix} Integrated Gradients", transitions)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def plot_first_5_matrices(tensor, title, n_time=5):
    # tensor shape: [T, C, C]
    transitions, labels = get_domain_boundaries()
    
    fig, axes = plt.subplots(1, n_time, figsize=(4*n_time, 4))
    fig.suptitle(title, fontsize=16)
    
    abs_max = np.max(np.abs(tensor[:n_time]))
    if abs_max == 0: abs_max = 1e-6
    vmin, vmax = -abs_max, abs_max
    
    for t in range(n_time):
        im = plot_domain_matrix(tensor[t], axes[t], f"Time {t}", transitions, vmin=vmin, vmax=vmax)
    
    # Add a single colorbar to the right
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
