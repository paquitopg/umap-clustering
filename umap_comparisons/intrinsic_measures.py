import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import umap

def compute_continuity(X, X_embedded, n_neighbors=5):
    """
    Calculates continuity: 
    Checks if neighbors in high-dim space are preserved in low-dim space.
    (Inverse logic of trustworthiness).
    """
    from sklearn.neighbors import NearestNeighbors
    n_samples = X.shape[0]
    
    # NN
    nn_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    high_ind = nn_high.kneighbors(X, return_distance=False)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded)
    low_ind = nn_low.kneighbors(X_embedded, return_distance=False)

    # Calculate rank overlap 
    # Score of 1.0 means perfect preservation of neighborhoods
    overlap = 0
    for i in range(n_samples):
        intersection = np.intersect1d(high_ind[i], low_ind[i])
        overlap += len(intersection)
    
    return overlap / (n_samples * n_neighbors)

print("Loading Data...")
fashion = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')
X = fashion.data.iloc[:3000].values  # Smaller subset for expensive distance matrix calcs

algorithms = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42),
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
}

# Store results for all methods
results = {}

# Compute embeddings and metrics for each method
for name, algo in algorithms.items():
    print(f"\n{'='*50}")
    print(f"Processing {name}...")
    print('='*50)
    
    print(f"Running {name}...")
    X_emb = algo.fit_transform(X)
    
    print("Computing Trustworthiness...")
    trust = trustworthiness(X, X_emb, n_neighbors=5)
    
    print("Computing Continuity...")
    cont = compute_continuity(X, X_emb, n_neighbors=5)
    
    print("-" * 30)
    print(f"Trustworthiness: {trust:.4f} (Avoids false neighbors)")
    print(f"Continuity:      {cont:.4f}  (Keeps neighbors together)")
    print("-" * 30)
    
    # Store results
    results[name] = {
        'embedding': X_emb,
        'trustworthiness': trust,
        'continuity': cont
    }

# Generate Shephard Diagrams for each method
print("\n" + "="*50)
print("Generating Shephard Diagrams...")
print("="*50)

# Randomly sample pairs to avoid memory explosion (O(N^2))
n_pairs = 5000
indices = np.random.choice(X.shape[0], 1000, replace=False)
X_sub = X[indices]

# Create subplot for all Shephard diagrams
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    X_emb_sub = result['embedding'][indices]
    
    # Pairwise distances
    dist_high = pairwise_distances(X_sub)
    dist_low = pairwise_distances(X_emb_sub)
    
    # Upper triangle
    high_vals = dist_high[np.triu_indices_from(dist_high, k=1)]
    low_vals = dist_low[np.triu_indices_from(dist_low, k=1)]
    
    # Compute Spearman correlation
    spearman_rho = spearmanr(high_vals, low_vals).correlation
    
    # Subsample points for plot
    mask = np.random.choice(len(high_vals), n_pairs, replace=False)
    
    ax = axes[idx]
    ax.scatter(high_vals[mask], low_vals[mask], alpha=0.3, s=5)
    ax.set_xlabel("Distance in High-Dim (Original)")
    ax.set_ylabel(f"Distance in Low-Dim ({name})")
    ax.set_title(f"{name} Shephard Diagram\nSpearman Rho: {spearman_rho:.3f}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Images/shephard_diagram.png")
plt.show()

# Print summary table
print("\n" + "="*50)
print("SUMMARY OF INTRINSIC MEASURES")
print("="*50)
print(f"{'Method':<10} {'Trustworthiness':<18} {'Continuity':<12} {'Spearman Rho':<15}")
print("-" * 50)
for name, result in results.items():
    # Compute Spearman for summary
    indices_summary = np.random.choice(X.shape[0], 1000, replace=False)
    X_sub_summary = X[indices_summary]
    X_emb_sub_summary = result['embedding'][indices_summary]
    dist_high_summary = pairwise_distances(X_sub_summary)
    dist_low_summary = pairwise_distances(X_emb_sub_summary)
    high_vals_summary = dist_high_summary[np.triu_indices_from(dist_high_summary, k=1)]
    low_vals_summary = dist_low_summary[np.triu_indices_from(dist_low_summary, k=1)]
    spearman_rho_summary = spearmanr(high_vals_summary, low_vals_summary).correlation
    
    print(f"{name:<10} {result['trustworthiness']:<18.4f} {result['continuity']:<12.4f} {spearman_rho_summary:<15.4f}")
print("="*50)