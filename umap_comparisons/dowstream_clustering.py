import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Configuration
n_points = 5000
dimensions = [1, 2, 3] 
n_clusters_kmeans = 10
eps_values = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

print("="*70)
print("LOADING DATA")
print("="*70)
fashion = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')
X = fashion.data.iloc[:n_points].values
y = fashion.target.iloc[:n_points].astype(int).values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# ============================================================================
# PHASE 1: GENERATE EMBEDDINGS FOR ALL DIMENSIONS
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: GENERATING EMBEDDINGS FOR ALL DIMENSIONS")
print("="*70)

embeddings = {} 

for dim in dimensions:
    print(f"\n--- Generating {dim}D embeddings ---")
    
    # Define algorithms for this dimension
    algorithms = {
        "PCA": PCA(n_components=dim, random_state=42),
        "t-SNE": TSNE(n_components=dim, perplexity=30, init='pca', 
                      learning_rate='auto', random_state=42),
        "UMAP": umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, random_state=42)
    }
    
    embeddings[dim] = {}
    
    for name, algo in algorithms.items():
   