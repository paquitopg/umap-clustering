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
        TSNE(n_components=dim, n_jobs=-1).fit_transform(X_sub)
        duration = time.time() - start
        results_dim['t-SNE']['dims'].append(dim)
        results_dim['t-SNE']['times'].append(duration)
        print(f"t-SNE: {duration:.2f}s")
    else:
        print("t-SNE: Skipped (dimension >= 4, Barnes Hut constraint)")

# ============================================================================
# PHASE 3: SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("PHASE 3: SAVING RESULTS")
print("="*70)

size_data = []
for method in ['PCA', 't-SNE', 'UMAP']:
    for size, time_val in zip(results_size[method]['sizes'], results_size[method]['times']):
        size_data.append({
            'Method': method,
            'Sample_Size': size,
            'Time_seconds': time_val,
            'Dimension': 2
        })

df_size = pd.DataFrame(size_data)
csv_size = f"images/computational_cost_sample_size.csv"
df_size.to_csv(csv_size, index=False)
print(f"Sample size scaling results saved to: {csv_size}")

dim_data = []
for method in ['PCA', 't-SNE', 'UMAP']:
    for dim, time_val in zip(results_dim[method]['dims'], results_dim[method]['times']):
        dim_data.append({
            'Method': method,
            'Dimension': dim,
            'Time_seconds': time_val,
            'Sample_Size': fixed_sample_size
        })

df_dim = pd.DataFrame(dim_data)
csv_dim = f"images/computational_cost_dimension.csv"
df_dim.to_csv(csv_dim, index=False)
print(f"Dimension scaling results saved to: {csv_dim}")

# ============================================================================
# PHASE 4: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("PHASE 4: GENERATING VISUALIZATIONS")
print("="*70)

print("Generating final embedding for visualization...")
embedding_final = umap.UMAP(n_neighbors=15, n_jobs=-1).fit_transform(X)

from matplotlib.lines import Line2D

# 1. Sample Size Scalability
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(results_size['PCA']['sizes'], results_size['PCA']['times'], 'o-', label='PCA', linewidth=2)
ax1.plot(results_size['UMAP']['sizes'], results_size['UMAP']['times'], 'o-', label='UMAP', linewidth=2)
if results_size['t-SNE']['sizes']:
    ax1.plot(results_size['t-SNE']['sizes'], results_size['t-SNE']['times'], 'x--', label='t-SNE', linewidth=2)
ax1.set_xlabel('Sample Size (N)', fontsize=12)
ax1.set_ylabel('Time (s)', fontsize=12)
ax1.set_yscale('log')
ax1.set_title('Scalability: Sample Size (Dimension = 2)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
filename1 = "images/scalability_sample_size.png"
plt.savefig(filename1, dpi=150, bbox_inches='tight')
print(f"Saved: {filename1}")
plt.close()

# 2. Dimension Scalability
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(results_dim['PCA']['dims'], results_dim['PCA']['times'], 'o-', label='PCA', linewidth=2)
ax2.plot(results_dim['UMAP']['dims'], results_dim['UMAP']['times'], 'o-', label='UMAP', linewidth=2)
if results_dim['t-SNE']['dims']:
    ax2.plot(results_dim['t-SNE']['dims'], results_dim['t-SNE']['times'], 'x--', label='t-SNE', linewidth=2)
ax2.set_xlabel('Embedding Dimension', fontsize=12)
ax2.set_ylabel('Time (s)', fontsize=12)
ax2.set_yscale('log')
ax2.set_title(f'Scalability: Embedding Dimension (N = {fixed_sample_size:,})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=4, color='red', linestyle='--', alpha=0.5)
ax2.text(4.2, ax2.get_ylim()[1]*0.5, 't-SNE\n< 4D only', 
         fontsize=10, color='red', verticalalignment='center')
plt.tight_layout()
filename2 = "images/scalability_dimension.png"
plt.savefig(filename2, dpi=150, bbox_inches='tight')
print(f"Saved: {filename2}")
plt.close()

# 3. Physics Separation
fig3, ax3 = plt.subplots(figsize=(8, 8))
scatter = ax3.scatter(embedding_final[:, 0], embedding_final[:, 1], 
                      c=y, cmap='coolwarm', s=0.1, alpha=0.5)
ax3.set_title(f'Mini-BooNE Separation (N={len(X):,})', fontsize=14, fontweight='bold')
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                         markersize=10, label='Background'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                         markersize=10, label='Signal')]
ax3.legend(handles=legend_elements, fontsize=11)
ax3.axis('off')
plt.tight_layout()
filename3 = "images/miniboone_separation.png"
plt.savefig(filename3, dpi=150, bbox_inches='tight')
print(f"Saved: {filename3}")
plt.close()

# 4. Comparison Table
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.axis('off')

table_data = []
for method in ['PCA', 'UMAP', 't-SNE']:
    if results_dim[method]['dims']:
        max_dim = max(results_dim[method]['dims'])
        max_dim_time = results_dim[method]['times'][results_dim[method]['dims'].index(max_dim)]
        table_data.append([method, f"{max_dim}D", f"{max_dim_time:.2f}s"])
    else:
        table_data.append([method, "N/A", "N/A"])

table = ax4.table(cellText=table_data,
                  colLabels=['Method', 'Max Dimension', 'Time (max dim)'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)
for i in range(len(table_data) + 1):
    for j in range(3):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
ax4.set_title('Dimension Capability Summary', fontsize=14, fontweight='bold', pad=30)
plt.tight_layout()
filename4 = "images/dimension_capability_summary.png"
plt.savefig(filename4, dpi=150, bbox_inches='tight')
print(f"Saved: {filename4}")
plt.close()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nSample Size Scaling (Dimension = 2):")
print(f"{'Method':<10} {'Max N':<12} {'Time at Max N':<15}")
print("-" * 70)
for method in ['PCA', 'UMAP', 't-SNE']:
    if results_size[method]['sizes']:
        max_n = max(results_size[method]['sizes'])
        max_n_idx = results_size[method]['sizes'].index(max_n)
        max_time = results_size[method]['times'][max_n_idx]
        print(f"{method:<10} {max_n:<12,} {max_time:<15.2f}s")
    else:
        print(f"{method:<10} {'N/A':<12} {'N/A':<15}")

print("\nDimension Scaling (N = {:,}):".format(fixed_sample_size))
print(f"{'Method':<10} {'Max Dim':<12} {'Time at Max Dim':<15}")
print("-" * 70)
for method in ['PCA', 'UMAP', 't-SNE']:
    if results_dim[method]['dims']:
        max_dim = max(results_dim[method]['dims'])
        max_dim_idx = results_dim[method]['dims'].index(max_dim)
        max_time = results_dim[method]['times'][max_dim_idx]
        print(f"{method:<10} {max_dim:<12} {max_time:<15.2f}s")
    else:
        print(f"{method:<10} {'N/A':<12} {'N/A':<15}")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)