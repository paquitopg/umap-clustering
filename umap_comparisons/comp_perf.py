import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("="*70)
print("LOADING DATA")
print("="*70)
print("Fetching Mini-BooNE dataset (Physics)...")

miniboone = fetch_openml(data_id=41150, as_frame=False, parser='auto')
print(f"Loaded Mini-BooNE dataset (ID: 41150)")

X = miniboone.data
y = miniboone.target

X = np.asarray(X)
y = np.asarray(y).ravel()  

X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)

print(f"Dataset Loaded. Shape: {X.shape}")

# Configuration
sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 130000]
dimensions = [2, 3, 4, 5, 6, 7, 10]  
fixed_sample_size = 25000  

# ============================================================================
# PHASE 1: SCALING WITH SAMPLE SIZE
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: SCALING WITH SAMPLE SIZE (Dimension = 2)")
print("="*70)

results_size = {
    'PCA':   {'sizes': [], 'times': []},
    't-SNE': {'sizes': [], 'times': []},
    'UMAP':  {'sizes': [], 'times': []}
}

for n in sample_sizes:
    if n > len(X): break
    
    print(f"\n--- Benchmarking N = {n} ---")
    
    indices = np.random.choice(len(X), n, replace=False)
    X_sub = X[indices]
    y_sub = y[indices]
    
    # PCA
    start = time.time()
    PCA(n_components=2).fit_transform(X_sub)
    duration = time.time() - start
    results_size['PCA']['sizes'].append(n)
    results_size['PCA']['times'].append(duration)
    print(f"PCA:   {duration:.2f}s")
    
    # UMAP
    start = time.time()
    umap.UMAP(n_neighbors=15, min_dist=0.1, n_jobs=-1).fit_transform(X_sub)
    duration = time.time() - start
    results_size['UMAP']['sizes'].append(n)
    results_size['UMAP']['times'].append(duration)
    print(f"UMAP:  {duration:.2f}s")
    
    # t-SNE
    if n <= 25000:
        start = time.time()
        TSNE(n_components=2, n_jobs=-1).fit_transform(X_sub)
        duration = time.time() - start
        results_size['t-SNE']['sizes'].append(n)
        results_size['t-SNE']['times'].append(duration)
        print(f"t-SNE: {duration:.2f}s")
    else:
        print("t-SNE: Skipped (Predicted > 10 mins)")

# ============================================================================
# PHASE 2: SCALING WITH EMBEDDING DIMENSION (Fixed sample size)
# ============================================================================
print("\n" + "="*70)
print(f"PHASE 2: SCALING WITH EMBEDDING DIMENSION (N = {fixed_sample_size})")
print("="*70)
print("Note: t-SNE only supports dimensions < 4 (Barnes Hut constraint)")
print("="*70)

if fixed_sample_size > len(X):
    fixed_sample_size = len(X)

indices = np.random.choice(len(X), fixed_sample_size, replace=False)
X_sub = X[indices]
y_sub = y[indices]

results_dim = {
    'PCA':   {'dims': [], 'times': []},
    't-SNE': {'dims': [], 'times': []},
    'UMAP':  {'dims': [], 'times': []}
}

for dim in dimensions:
    print(f"\n--- Benchmarking Dimension = {dim} ---")
    
    # PCA
    start = time.time()
    PCA(n_components=dim).fit_transform(X_sub)
    duration = time.time() - start
    results_dim['PCA']['dims'].append(dim)
    results_dim['PCA']['times'].append(duration)
    print(f"PCA:   {duration:.2f}s")
    
    # UMAP
    start = time.time()
    umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, n_jobs=-1).fit_transform(X_sub)
    duration = time.time() - start
    results_dim['UMAP']['dims'].append(dim)
    results_dim['UMAP']['times'].append(duration)
    print(f"UMAP:  {duration:.2f}s")
    
    # t-SNE
    if dim < 4:
        start = time.time()
        TSNE(n_components=dim, n_jobs=-1).fit