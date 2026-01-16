import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from umap_algo.umap_class import umap_mapping
import umap  # UMAP library for comparison

print("="*70)
print("TESTING CUSTOM UMAP ON FOREST COVER DATASET")
print("="*70)

# Load Forest Cover dataset
print("\nLoading Forest Cover dataset...")
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Normalize the data
print("\nNormalizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use a subset:
n_points = 5000
indices = np.random.choice(len(X_scaled), n_points, replace=False)
X_scaled = X_scaled[indices]
y = y[indices]
print(f"Using subset: {n_points} points (for faster testing)")

# Initialize both UMAP implementations
print("\nInitializing UMAP implementations...")
print("Parameters: n_neighbors=15, n_components=2, min_dist=0.1")

# library
umap_library = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

# Custom
umap_custom = umap_mapping(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    KNN_method='approx',  # 'exact' or 'approx'
    random_state=42
)

print("\nRunning UMAP library...")
start_lib = time.time()
embedding_lib = umap_library.fit_transform(X_scaled)
duration_lib = time.time() - start_lib
print(f"UMAP library completed in {duration_lib:.2f} seconds")

print("\nRunning custom UMAP implementation (this may take a while)...")
start_custom = time.time()
embedding_custom = umap_custom.fit_transform(X_scaled, n_epochs=200)
duration_custom = time.time() - start_custom
print(f"Custom UMAP completed in {duration_custom:.2f} seconds")

print("\nGenerating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# left
ax1 = axes[0]
scatter1 = ax1.scatter(
    embedding_lib[:, 0],
    embedding_lib[:, 1],
    c=y,
    cmap='Spectral',
    s=5,
    alpha=0.6
)
ax1.set_title(
    f"UMAP Library\n"
    f"Time: {duration_lib:.2f}s",
    fontsize=14,
    fontweight='bold'
)
ax1.set_xlabel("UMAP Dimension 1", fontsize=12)
ax1.set_ylabel("UMAP Dimension 2", fontsize=12)
ax1.grid(True, alpha=0.3)

# right
ax2 = axes[1]
scatter2 = ax2.scatter(
    embedding_custom[:, 0],
    embedding_custom[:, 1],
    c=y,
    cmap='Spectral',
    s=5,
    alpha=0.6
)
ax2.set_title(
    f"Custom UMAP Implementation\n"
    f"Time: {duration_custom:.2f}s",
    fontsize=14,
    fontweight='bold'
)
ax2.set_xlabel("UMAP Dimension 1", fontsize=12)
ax2.set_ylabel("UMAP Dimension 2", fontsize=12)
ax2.grid(True, alpha=0.3)

# title
plt.suptitle(
    f"UMAP Comparison: Forest Cover Dataset ({len(X_scaled):,} points)",
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# layout
plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# colorbar
cbar = fig.colorbar(scatter1, ax=axes, orientation='horizontal', 
                    fraction=0.05, pad=0.15, aspect=40)
cbar.set_label('Forest Cover Type', fontsize=12, fontweight='bold')

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.j