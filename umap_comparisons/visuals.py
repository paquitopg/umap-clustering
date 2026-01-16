import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time

print("Loading Fashion-MNIST...")

fashion = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')
n_points = 25000
X = fashion.data.iloc[:n_points].values
y = fashion.target.iloc[:n_points].astype(int).values
target_names = fashion.target_names if hasattr(fashion, 'target_names') else [str(i) for i in range(10)]

algorithms = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42),
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (name, algo) in enumerate(algorithms.items()):
    print(f"Running {name}...")
    start = time.time()
    embedding = algo.fit_transform(X)
    duration = time.time() - start
    
    ax = axes[i]
    scatter = ax.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=y, 
        cmap='Spectral', 
        s=5, 
        alpha=0.7
    )
    ax.set_title(f"{name} (Time: {duration:.2f}s)", fontsize=14)
    ax.axis('off')

# Add a single colorbar
cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
cbar.set_label('Digit Class')
plt.suptitle(f"Qualitative Comparison: Fashion-MNIST Embeddings ({n_points} points)", fontsize=18)
plt.savefig(f"images/qualitative_comparison_{n_points}points.png")
plt.show()