import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

#TEST

#SIZES = [5000, 20000, 50000, 100000, 500000] # Full scale
SIZES = [1000, 5000, 10000] # Small scale for testing
TIMEOUT = 300 # 5 minutes max per run

# Load Data (Assuming data is loaded into X_full as before)
# For testing without the file, we simulate:
X_full = np.random.rand(100000, 5) 

results = {
    'PCA': {'sizes': [], 'times': []},
    't-SNE': {'sizes': [], 'times': []},
    'UMAP (Lib)': {'sizes': [], 'times': []}
}

def run_with_timeout(algo_name, algo_func, X, sizes_list):
    """Runs the algorithm for increasing sizes until it times out."""
    print(f"\n--- Benchmarking {algo_name} ---")
    for n in sizes_list:
        if n > len(X): break
        
        X_sub = X[:n]
        start = time.time()
        
        try:
            algo_func(X_sub)
            duration = time.time() - start
            
            print(f"Size {n}: {duration:.2f}s")
            results[algo_name]['sizes'].append(n)
            results[algo_name]['times'].append(duration)
            
            if duration > TIMEOUT:
                print(f"-> Stopping {algo_name} (Time Limit Exceeded)")
                break
                
        except Exception as e:
            print(f"-> {algo_name} FAILED at N={n}: {e}")
            break

# 1. Define Wrappers
run_with_timeout('PCA', 
                 lambda data: PCA(n_components=2).fit_transform(data), 
                 X_full, SIZES)

run_with_timeout('UMAP (Lib)', 
                 lambda data: umap.UMAP(n_neighbors=15, n_jobs=-1).fit_transform(data), 
                 X_full, SIZES)

run_with_timeout('t-SNE', 
                 lambda data: TSNE(n_components=2, n_jobs=-1).fit_transform(data), 
                 X_full, SIZES)

# 2. Plot
plt.figure(figsize=(10, 6))
for name, res in results.items():
    if res['sizes']:
        plt.plot(res['sizes'], res['times'], marker='o', label=name)

plt.xlabel('Sample Size (N)')
plt.ylabel('Time (Seconds)')
plt.title('Scalability: NYC Taxi Dataset')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig("Images/comp_perf.png")
