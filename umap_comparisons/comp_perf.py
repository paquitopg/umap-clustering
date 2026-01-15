import time
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.utils import resample


print("Fetching Forest Covertype dataset (this may take a moment)...")
covtype = fetch_covtype()
X_full = covtype.data
y_full = covtype.target

print(f"Dataset Loaded. Shape: {X_full.shape}")

# We define increasing sizes to test the O(N) complexity
sizes = [5000, 10000, 50000, 100000] # Stopped at 100k for t-SNE safety. 

results = {
    'PCA':   {'sizes': [], 'times': [], 'score': []},
    't-SNE': {'sizes': [], 'times': [], 'score': []},
    'UMAP':  {'sizes': [], 'times': [], 'score': []}
}

for n in sizes:
    print(f"\n--- Benchmarking N = {n} ---")
    
    # Stratified subsample to keep class balance
    X_sub, y_sub = resample(X_full, y_full, n_samples=n, random_state=42, stratify=y_full)
    
    # PCA
    start = time.time()
    embedding_pca = PCA(n_components=2).fit_transform(X_sub)
    pca_time = time.time() - start
    results['PCA']['sizes'].append(n)
    results['PCA']['times'].append(pca_time)
    print(f"PCA:   {pca_time:.2f}s")
    
    # UMAP
    start = time.time()
    # n_jobs=-1 uses all cores. 
    embedding_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_jobs=-1).fit_transform(X_sub)
    umap_time = time.time() - start
    results['UMAP']['sizes'].append(n)
    results['UMAP']['times'].append(umap_time)
    print(f"UMAP:  {umap_time:.2f}s")
    
    # t-SNE
    if n <= 50000:
        start = time.time()
        embedding_tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(X_sub)
        tsne_time = time.time() - start
        results['t-SNE']['sizes'].append(n)
        results['t-SNE']['times'].append(tsne_time)
        print(f"t-SNE: {tsne_time:.2f}s")
    else:
        print("t-SNE: Skipped (Too slow for this size)")

# Quality Check 
print("\n--- Quality Sanity Check (Silhouette Score on N=50,000) ---")
idx = np.random.choice(len(X_sub), 10000, replace=False)

sil_pca = silhouette_score(embedding_pca[idx], y_sub[idx])
sil_umap = silhouette_score(embedding_umap[idx], y_sub[idx])
print(f"PCA Silhouette:  {sil_pca:.3f}")
print(f"UMAP Silhouette: {sil_umap:.3f}")

plt.figure(figsize=(12, 5))

# Plot 1: Scalability (Time)
plt.subplot(1, 2, 1)
plt.plot(results['PCA']['sizes'], results['PCA']['times'], 'o-', label='PCA')
plt.plot(results['UMAP']['sizes'], results['UMAP']['times'], 'o-', label='UMAP')
if results['t-SNE']['sizes']:
    plt.plot(results['t-SNE']['sizes'], results['t-SNE']['times'], 'x--', label='t-SNE')

plt.xlabel('Sample Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Computational Cost')
plt.legend()
plt.yscale('log') # Log scale is crucial here
plt.grid(True, alpha=0.3)

# Plot 2: Visual Result
plt.subplot(1, 2, 2)
plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=y_sub, cmap='Spectral', s=0.1, alpha=0.5)
plt.title(f'UMAP Visualization (N={n})')
plt.axis('off')

plt.tight_layout()
plt.savefig("Images/performance_covtype.png")
plt.show()