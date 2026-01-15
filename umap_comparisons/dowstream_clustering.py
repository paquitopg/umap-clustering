import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

fashion = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')
X = fashion.data.iloc[:5000].values
y = fashion.target.iloc[:5000].astype(int).values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

algorithms = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42),
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
}

# Store embeddings and results
embeddings = {}
results = {}

print("="*60)
print("Generating embeddings for all methods...")
print("="*60)

# Generate embeddings for each method
for name, algo in algorithms.items():
    print(f"\nProjecting data to 2D with {name}...")
    X_emb = algo.fit_transform(X_scaled)
    embeddings[name] = X_emb

print("\n" + "="*60)
print("Evaluating K-Means Clustering (K=10)")
print("="*60)

# K-Means evaluation for all methods
kmeans_results = {}
for name, X_emb in embeddings.items():
    print(f"\n--- {name} ---")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X_emb)
    
    ari_kmeans = adjusted_rand_score(y, labels_pred)
    sil_kmeans = silhouette_score(X_emb, labels_pred)
    
    print(f"ARI Score: {ari_kmeans:.4f} (1.0 is perfect match with Truth)")
    print(f"Silhouette: {sil_kmeans:.4f} (Higher is better separation)")
    
    kmeans_results[name] = {
        'ari': ari_kmeans,
        'silhouette': sil_kmeans,
        'labels': labels_pred
    }

print("\n" + "="*60)
print("Evaluating DBSCAN Clustering")
print("="*60)
print("Note: Embeddings are normalized to the same scale before DBSCAN")
print("      to ensure fair comparison with the same eps parameter.")
print("="*60)

# Hyperparameter optimization for eps
# Test a range of eps values
eps_values = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

# Store all results for each eps
all_dbscan_results = {}

# Normalize embeddings once (same scaler for all eps values)
normalized_embeddings = {}
for name, X_emb in embeddings.items():
    normalized_embeddings[name] = scaler.fit_transform(X_emb)

# Test each eps value
for eps in eps_values:
    print(f"\n{'='*60}")
    print(f"Testing eps = {eps:.2f}")
    print('='*60)
    
    dbscan_results = {}
    
    for name, X_emb_normalized in normalized_embeddings.items():
        print(f"\n--- {name} ---")
        
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels_db = dbscan.fit_predict(X_emb_normalized)
        
        # Noise points (-1) for evaluation
        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise = np.sum(labels_db == -1)
        
        if len(set(labels_db)) > 1:
            ari_db = adjusted_rand_score(y, labels_db)
            sil_db = silhouette_score(X_emb_normalized, labels_db)
            print(f"ARI Score: {ari_db:.4f}")
            print(f"Silhouette: {sil_db:.4f}")
            print(f"Clusters found: {n_clusters}")
            print(f"Noise points: {n_noise}")
            
            dbscan_results[name] = {
                'ari': ari_db,
                'silhouette': sil_db,
                'labels': labels_db,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
        else:
            print(f"DBSCAN found only noise or one cluster.")
            dbscan_results[name] = {
                'ari': None,
                'silhouette': None,
                'labels': labels_db,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
    
    # Store results for this eps
    all_dbscan_results[eps] = dbscan_results
    
    # Print summary table for this eps
    print(f"\n{'='*60}")
    print(f"SUMMARY: DBSCAN Clustering Comparison (eps = {eps:.2f})")
    print("="*60)
    print(f"{'Method':<10} {'ARI Score':<15} {'Silhouette Score':<18} {'Clusters':<10} {'Noise':<10}")
    print("-" * 60)
    for name, result in dbscan_results.items():
        ari_str = f"{result['ari']:.4f}" if result['ari'] is not None else "N/A"
        sil_str = f"{result['silhouette']:.4f}" if result['silhouette'] is not None else "N/A"
        print(f"{name:<10} {ari_str:<15} {sil_str:<18} {result['n_clusters']:<10} {result['n_noise']:<10}")
    print("="*60)

# Print summary comparison for K-Means
print("\n" + "="*60)
print("SUMMARY: K-Means Clustering Comparison")
print("="*60)
print(f"{'Method':<10} {'ARI Score':<15} {'Silhouette Score':<18}")
print("-" * 60)
for name, result in kmeans_results.items():
    print(f"{name:<10} {result['ari']:<15.4f} {result['silhouette']:<18.4f}")
print("="*60)

# Find best eps based on average ARI score across all methods
print("\n" + "="*60)
print("EPS OPTIMIZATION SUMMARY")
print("="*60)
print(f"{'eps':<10} {'Avg ARI':<15} {'Avg Silhouette':<18} {'Best Method':<15}")
print("-" * 60)

best_eps = None
best_avg_ari = -1

for eps in eps_values:
    results = all_dbscan_results[eps]
    ari_scores = [r['ari'] for r in results.values() if r['ari'] is not None]
    sil_scores = [r['silhouette'] for r in results.values() if r['silhouette'] is not None]
    
    if ari_scores:
        avg_ari = np.mean(ari_scores)
        avg_sil = np.mean(sil_scores) if sil_scores else None
        
        # Find best method for this eps
        best_method = max(results.items(), key=lambda x: x[1]['ari'] if x[1]['ari'] is not None else -1)[0]
        
        avg_sil_str = f"{avg_sil:.4f}" if avg_sil is not None else "N/A"
        print(f"{eps:<10.2f} {avg_ari:<15.4f} {avg_sil_str:<18} {best_method:<15}")
        
        if avg_ari > best_avg_ari:
            best_avg_ari = avg_ari
            best_eps = eps
    else:
        print(f"{eps:<10.2f} {'N/A':<15} {'N/A':<18} {'N/A':<15}")

if best_eps is not None:
    print("\n" + "="*60)
    print(f"BEST EPS: {best_eps:.2f} (Average ARI: {best_avg_ari:.4f})")
    print("="*60)