import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def preprocess_matrices(data, threshold_k_percent=None, binarize=False):
    """
    Preprocess matrices:
    1. Take absolute values
    2. Nullify diagonals
    3. (Optional) Threshold to keep only top K% values per matrix
    4. (Optional) Binarize all non-zero values to 1
    """
    res = np.abs(data.copy())
    
    # Store original shape to reshape labels later if needed
    orig_shape = res.shape
    
    if res.ndim == 2:
        res = res[np.newaxis, ...]
    elif res.ndim > 3:
        # e.g., [B, T, C, C] -> [B*T, C, C]
        res = res.reshape(-1, res.shape[-2], res.shape[-1])
        
    N, C, _ = res.shape
    
    # Nullify diagonal for all matrices
    mask = np.eye(C, dtype=bool)
    res[..., mask] = 0.0
    
    if threshold_k_percent is not None:
        flat = res.reshape(N, -1)
        p = 100 - threshold_k_percent
        thresholds = np.percentile(flat, p, axis=1, keepdims=True)
        flat[flat < thresholds] = 0.0
        res = flat.reshape(N, C, C)
        
    if binarize:
        res[res > 0] = 1.0
        
    # Squeeze back if input was 2D
    if data.ndim == 2:
        return res[0]
    return res

def sort_clusters_by_support(labels, *centroids_args):
    """
    Renames cluster labels so that 0 is the most frequent, 1 is the next, etc.
    """
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    sorted_unique = unique[sorted_idx]
    
    mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_unique)}
    new_labels = np.array([mapping[l] for l in labels])
    
    if len(centroids_args) > 0:
        if len(centroids_args) == 1 and centroids_args[0] is None:
            return new_labels
            
        sorted_centroids = [c[sorted_idx] if c is not None else None for c in centroids_args]
        if len(sorted_centroids) == 1:
            return new_labels, sorted_centroids[0]
        return (new_labels, *sorted_centroids)
    return new_labels

def plot_elbow_criterion(data, k_max=10, n_init=10, threshold_k_percent=None, step=1):
    processed = preprocess_matrices(data, threshold_k_percent)
    flat = processed.reshape(processed.shape[0], -1)
    
    inertias = []
    ks = list(range(2, k_max + 1, step))
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        kmeans.fit(flat)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.title("Elbow Criterion")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

def plot_silhouette_scores(data, k_max=10, n_init=10, threshold_k_percent=None, step=1):
    processed = preprocess_matrices(data, threshold_k_percent)
    flat = processed.reshape(processed.shape[0], -1)
    
    scores = []
    ks = list(range(2, k_max + 1, step))
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        labels = kmeans.fit_predict(flat)
        score = silhouette_score(flat, labels)
        scores.append(score)
        
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o')
    plt.title("Silhouette Scores")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.grid(True)
    plt.show()

def plot_elbow_criterion_bin(data, k_max=10, n_init=10, threshold_k_percent=5, step=1):
    processed = preprocess_matrices(data, threshold_k_percent=threshold_k_percent, binarize=True)
    flat = processed.reshape(processed.shape[0], -1)
    
    inertias = []
    ks = list(range(2, k_max + 1, step))
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        kmeans.fit(flat)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.title("Elbow Criterion (Binarized)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

def plot_silhouette_scores_bin(data, k_max=10, n_init=10, threshold_k_percent=5, step=1):
    processed = preprocess_matrices(data, threshold_k_percent=threshold_k_percent, binarize=True)
    flat = processed.reshape(processed.shape[0], -1)
    
    scores = []
    ks = list(range(2, k_max + 1, step))
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        labels = kmeans.fit_predict(flat)
        try:
            score = silhouette_score(flat, labels)
        except ValueError:
            score = 0
        scores.append(score)
        
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o')
    plt.title("Silhouette Scores (Binarized)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.grid(True)
    plt.show()

def robust_kmeans(data, n_clusters, n_init=20, threshold_k_percent=None):
    processed = preprocess_matrices(data, threshold_k_percent)
    N, C, _ = processed.shape
    flat = processed.reshape(N, -1)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(flat)
    
    # Calculate centroids using the processed arrays
    centroids = kmeans.cluster_centers_.reshape(n_clusters, C, C)
    
    labels, centroids = sort_clusters_by_support(labels, centroids)
    
    # Reshape labels back to match the original data's leading dimensions (e.g. [B, T])
    if data.ndim > 3:
        labels = labels.reshape(data.shape[:-2])
        
    return labels, centroids, kmeans

def consensus_clustering(data, n_clusters, n_ensembles=100, threshold_k_percent=None):
    processed = preprocess_matrices(data, threshold_k_percent)
    N, C, _ = processed.shape
    flat = processed.reshape(N, -1)
    
    co_occurrence = np.zeros((N, N))
    
    for i in range(n_ensembles):
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=i)
        labels = kmeans.fit_predict(flat)
        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            xx, yy = np.meshgrid(idx, idx)
            co_occurrence[xx, yy] += 1
            
    co_occurrence /= n_ensembles
    dist_matrix = 1.0 - co_occurrence
    np.fill_diagonal(dist_matrix, 0.0)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    final_labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    
    centroids = np.zeros((n_clusters, C*C))
    for i in range(n_clusters):
        if np.sum(final_labels == i) > 0:
            centroids[i] = flat[final_labels == i].mean(axis=0)
    centroids = centroids.reshape(n_clusters, C, C)
        
    final_labels, centroids = sort_clusters_by_support(final_labels, centroids)
    
    # Reshape labels back to match the original data's leading dimensions (e.g. [B, T])
    if data.ndim > 3:
        final_labels = final_labels.reshape(data.shape[:-2])
    
    return final_labels, centroids, co_occurrence

def robust_kmeans_bin(data, n_clusters, n_init=20, threshold_k_percent=5):
    # Process original data without binarization for mapping back later
    processed_orig = preprocess_matrices(data, threshold_k_percent=None, binarize=False)
    
    # Process data with thresholding and binarization for clustering
    processed_bin = preprocess_matrices(data, threshold_k_percent=threshold_k_percent, binarize=True)
    
    N, C, _ = processed_bin.shape
    flat_bin = processed_bin.reshape(N, -1)
    flat_orig = processed_orig.reshape(N, -1)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(flat_bin)
    
    # Centroids of the binary matrices
    centroids_bin = kmeans.cluster_centers_.reshape(n_clusters, C, C)
    
    # Map from binary attribution back to original matrices
    centroids_orig = np.zeros((n_clusters, C, C))
    for i in range(n_clusters):
        if np.sum(labels == i) > 0:
            centroids_orig[i] = flat_orig[labels == i].mean(axis=0).reshape(C, C)
            
    labels, centroids_bin, centroids_orig = sort_clusters_by_support(labels, centroids_bin, centroids_orig)
    
    if data.ndim > 3:
        labels = labels.reshape(data.shape[:-2])
        
    return labels, centroids_bin, centroids_orig, kmeans

def consensus_clustering_bin(data, n_clusters, n_ensembles=100, threshold_k_percent=5):
    processed_orig = preprocess_matrices(data, threshold_k_percent=None, binarize=False)
    processed_bin = preprocess_matrices(data, threshold_k_percent=threshold_k_percent, binarize=True)
    
    N, C, _ = processed_bin.shape
    flat_bin = processed_bin.reshape(N, -1)
    flat_orig = processed_orig.reshape(N, -1)
    
    co_occurrence = np.zeros((N, N))
    
    for i in range(n_ensembles):
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=i)
        labels = kmeans.fit_predict(flat_bin)
        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            xx, yy = np.meshgrid(idx, idx)
            co_occurrence[xx, yy] += 1
            
    co_occurrence /= n_ensembles
    dist_matrix = 1.0 - co_occurrence
    np.fill_diagonal(dist_matrix, 0.0)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    final_labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    
    centroids_bin = np.zeros((n_clusters, C*C))
    centroids_orig = np.zeros((n_clusters, C*C))
    for i in range(n_clusters):
        if np.sum(final_labels == i) > 0:
            centroids_bin[i] = flat_bin[final_labels == i].mean(axis=0)
            centroids_orig[i] = flat_orig[final_labels == i].mean(axis=0)
            
    centroids_bin = centroids_bin.reshape(n_clusters, C, C)
    centroids_orig = centroids_orig.reshape(n_clusters, C, C)
        
    final_labels, centroids_bin, centroids_orig = sort_clusters_by_support(final_labels, centroids_bin, centroids_orig)
    
    if data.ndim > 3:
        final_labels = final_labels.reshape(data.shape[:-2])
    
    return final_labels, centroids_bin, centroids_orig, co_occurrence
