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
        print(f"  {name}...", end=" ", flush=True)
        if dim > 3 and name == "t-SNE":
            print(f"Warning: t-SNE for {dim}D may be slow", end=" ", flush=True)
        X_emb = algo.fit_transform(X_scaled)
        embeddings[dim][name] = X_emb
        print(f"Done. Shape: {X_emb.shape}")

# ============================================================================
# PHASE 2: HYPERPARAMETER TUNING (Find best eps for DBSCAN)
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: HYPERPARAMETER TUNING (DBSCAN eps optimization)")
print("="*70)
print("Note: Using 2D embeddings for hyperparameter tuning")
print("="*70)

tuning_embeddings = embeddings[2]
normalized_tuning_embeddings = {}
scaler_emb = StandardScaler()
for name, X_emb in tuning_embeddings.items():
    normalized_tuning_embeddings[name] = scaler_emb.fit_transform(X_emb)

tuning_results = {}

for eps in eps_values:
    print(f"\nTesting eps = {eps:.2f}")
    
    eps_results = {}
    for name, X_emb_normalized in normalized_tuning_embeddings.items():
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels_db = dbscan.fit_predict(X_emb_normalized)
        
        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise = np.sum(labels_db == -1)
        
        if len(set(labels_db)) > 1:
            ari_db = adjusted_rand_score(y, labels_db)
            eps_results[name] = ari_db
        else:
            eps_results[name] = -1  # Invalid result
    
    tuning_results[eps] = eps_results

print("\n" + "="*70)
print("EPS TUNING RESULTS")
print("="*70)
print(f"{'eps':<10} {'PCA ARI':<12} {'t-SNE ARI':<12} {'UMAP ARI':<12} {'Avg ARI':<12}")
print("-" * 70)

best_eps = None
best_avg_ari = -1
eps_summary = []

for eps in eps_values:
    results = tuning_results[eps]
    ari_scores = [r for r in results.values() if r >= 0]
    
    if ari_scores:
        avg_ari = np.mean(ari_scores)
        pca_ari = results.get('PCA', -1)
        tsne_ari = results.get('t-SNE', -1)
        umap_ari = results.get('UMAP', -1)
        
        pca_str = f"{pca_ari:.4f}" if pca_ari >= 0 else "N/A"
        tsne_str = f"{tsne_ari:.4f}" if tsne_ari >= 0 else "N/A"
        umap_str = f"{umap_ari:.4f}" if umap_ari >= 0 else "N/A"
        
        print(f"{eps:<10.2f} {pca_str:<12} {tsne_str:<12} {umap_str:<12} {avg_ari:<12.4f}")
        
        eps_summary.append({
            'eps': eps,
            'PCA_ARI': pca_ari if pca_ari >= 0 else None,
            't-SNE_ARI': tsne_ari if tsne_ari >= 0 else None,
            'UMAP_ARI': umap_ari if umap_ari >= 0 else None,
            'Avg_ARI': avg_ari
        })
        
        if avg_ari > best_avg_ari:
            best_avg_ari = avg_ari
            best_eps = eps
    else:
        print(f"{eps:<10.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

if best_eps is not None:
    print("\n" + "="*70)
    print(f"BEST EPS: {best_eps:.2f} (Average ARI: {best_avg_ari:.4f})")
    print("="*70)

df_eps_tuning = pd.DataFrame(eps_summary)
df_eps_tuning.to_csv("images/dbscan_eps_tuning.csv", index=False)
print(f"\nEPS tuning results saved to: images/dbscan_eps_tuning.csv")

# ============================================================================
# PHASE 3: EVALUATION (Run clustering with best eps and K-Means)
# ============================================================================
print("\n" + "="*70)
print("PHASE 3: CLUSTERING EVALUATION")
print("="*70)
print(f"Using best eps = {best_eps:.2f} for DBSCAN")
print("="*70)

all_results = []

for dim in dimensions:
    print(f"\n--- Evaluating {dim}D embeddings ---")
    
    dim_embeddings = embeddings[dim]
    
    normalized_dim_embeddings = {}
    scaler_dim = StandardScaler()
    for name, X_emb in dim_embeddings.items():
        normalized_dim_embeddings[name] = scaler_dim.fit_transform(X_emb)
    
    for name, X_emb in dim_embeddings.items():
        print(f"  {name}...")
        
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_emb)
        ari_kmeans = adjusted_rand_score(y, labels_kmeans)
        sil_kmeans = silhouette_score(X_emb, labels_kmeans)
        
        if name in normalized_dim_embeddings:
            dbscan = DBSCAN(eps=best_eps, min_samples=10)
            labels_dbscan = dbscan.fit_predict(normalized_dim_embeddings[name])
            
            n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
            n_noise = np.sum(labels_dbscan == -1)
            
            if len(set(labels_dbscan)) > 1:
                ari_dbscan = adjusted_rand_score(y, labels_dbscan)
                sil_dbscan = silhouette_score(normalized_dim_embeddings[name], labels_dbscan)
            else:
                ari_dbscan = None
                sil_dbscan = None
        else:
            ari_dbscan = None
            sil_dbscan = None
            n_clusters_db = None
            n_noise = None
        
        all_results.append({
            'Method': name,
            'Dimension': dim,
            'K-Means_ARI': ari_kmeans,
            'K-Means_Silhouette': sil_kmeans,
            'DBSCAN_ARI': ari_dbscan,
            'DBSCAN_Silhouette': sil_dbscan,
            'DBSCAN_Clusters': n_clusters_db,
            'DBSCAN_Noise': n_noise,
            'eps': best_eps
        })

# ============================================================================
# PHASE 4: SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("PHASE 4: SAVING RESULTS")
print("="*70)

df_results = pd.DataFrame(all_results)

csv_filename = f"images/downstream_clustering_results_{n_points}.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to: {csv_filename}")

txt_filename = f"images/downstream_clustering_results_{n_points}.txt"
with open(txt_filename, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DOWNSTREAM CLUSTERING EVALUATION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: Fashion-MNIST\n")
    f.write(f"Number of points: {n_points}\n")
    f.write(f"Dimensions tested: {dimensions}\n")
    f.write(f"K-Means clusters: {n_clusters_kmeans}\n")
    f.write(f"DBSCAN eps (tuned): {best_eps:.2f}\n")
    f.write("="*80 + "\n\n")
    
    for dim in dimensions:
        f.write(f"\n{'='*80}\n")
        f.write(f"DIMENSION {dim}D\n")
        f.write(f"{'='*80}\n\n")
        
        dim_results = df_results[df_results['Dimension'] == dim]
        
        f.write("K-Means Clustering:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<10} {'ARI Score':<15} {'Silhouette Score':<18}\n")
        f.write("-" * 80 + "\n")
        for _, row in dim_results.iterrows():
            f.write(f"{row['Method']:<10} {row['K-Means_ARI']:<15.4f} {row['K-Means_Silhouette']:<18.4f}\n")
        f.write("\n")
        
        f.write("DBSCAN Clustering:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<10} {'ARI Score':<15} {'Silhouette Score':<18} {'Clusters':<10} {'Noise':<10}\n")
        f.write("-" * 80 + "\n")
        for _, row in dim_results.iterrows():
            ari_str = f"{row['DBSCAN_ARI']:.4f}" if pd.notna(row['DBSCAN_ARI']) else "N/A"
            sil_str = f"{row['DBSCAN_Silhouette']:.4f}" if pd.notna(row['DBSCAN_Silhouette']) else "N/A"
            clusters_str = f"{int(row['DBSCAN_Clusters'])}" if pd.notna(row['DBSCAN_Clusters']) else "N/A"
            noise_str = f"{int(row['DBSCAN_Noise'])}" if pd.notna(row['DBSCAN_Noise']) else "N/A"
            f.write(f"{row['Method']:<10} {ari_str:<15} {sil_str:<18} {clusters_str:<10} {noise_str:<10}\n")
        f.write("\n")
    
    f.write("="*80 + "\n")

print(f"Formatted results saved to: {txt_filename}")

print("\n" + "="*70)
print("SUMMARY: K-Means Results")
print("="*70)
print(f"{'Method':<10} {'Dim':<6} {'ARI':<12} {'Silhouette':<12}")
print("-" * 70)
for _, row in df_results.iterrows():
    print(f"{row['Method']:<10} {row['Dimension']:<6} {row['K-Means_ARI']:<12.4f} {row['K-Means_Silhouette']:<12.4f}")

print("\n" + "="*70)
print("SUMMARY: DBSCAN Results")
print("="*70)
print(f"{'Method':<10} {'Dim':<6} {'ARI':<12} {'Silhouette':<12} {'Clusters':<10} {'Noise':<10}")
print("-" * 70)
for _, row in df_results.iterrows():
    ari_str = f"{row['DBSCAN_ARI']:.4f}" if pd.notna(row['DBSCAN_ARI']) else "N/A"
    sil_str = f"{row['DBSCAN_Silhouette']:.4f}" if pd.notna(row['DBSCAN_Silhouette']) else "N/A"
    clusters_str = f"{int(row['DBSCAN_Clusters'])}" if pd.notna(row['DBSCAN_Clusters']) else "N/A"
    noise_str = f"{int(row['DBSCAN_Noise'])}" if pd.notna(row['DBSCAN_Noise']) else "N/A"
    print(f"{row['Method']:<10} {row['Dimension']:<6} {ari_str:<12} {sil_str:<12} {clusters_str:<10} {noise_str:<10}")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)