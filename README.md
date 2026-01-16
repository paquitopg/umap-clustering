# UMAP-clustering

UMAP-Clustering is a project that applies Uniform Manifold Approximation and Projection (UMAP) — a modern non-linear dimensionality reduction algorithm — to real-world datasets and combines it with clustering techniques to uncover structure in high-dimensional data.

UMAP is known for producing meaningful low-dimensional embeddings that preserve local and some global structure of the original data, making it useful not just for visualization but also as a preprocessing step before clustering.


## 🚀 Features

### 📉 Dimensionality Reduction
Uses UMAP to reduce high-dimensional data into a lower-dimensional space where structure can be more easily visualized. UMAP algorithm can be taken from this repository or from the library `umap-learn`.

### 🔍 Clustering Analysis
Applies clustering algorithms (e.g., K-Means, DBSCAN) on UMAP embeddings to identify meaningful groups in the data.

### 📊 Comparison with Other Techniques
Enables comparison of UMAP results with classical methods like PCA or t-SNE.

## 📦 Project Structure
| Folder              | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `data_exploration/` | Scripts and notebooks for inspecting datasets.                          |
| `umap_algo/`        | Core UMAP implementation and embedding pipelines.                       |
| `umap_comparisons/` | Comparisons between UMAP and other dimensionality reduction approaches. |

## 🧠 About UMAP

UMAP (Uniform Manifold Approximation and Projection) is a method that:

* Constructs a graph of nearest neighbors in high dimension.

* Optimizes a low-dimensional embedding that preserves this structure.

* Is widely used for visualization and as a preprocessing step for clustering and other tasks.

## 📊 Typical Workflow

### 1. Load high-dimensional dataset
 Prepare your dataset with relevant features.

### 2. Dimensionality Reduction via UMAP
Reduce to a lower dimension (e.g., 2D or 10D) while preserving structure.

### 3. Apply Clustering
Run clustering algorithms such as K-Means, DBSCAN, or density-based methods over the UMAP output.

### 4. Evaluate Results
Use clustering metrics (e.g., silhouette score) to compare and validate performance.


## 📌 Example Use

Below is a minimal Python snippet demonstrating how to run UMAP and a basic clustering:

```
from umap_algo.umap_class import umap_mapping

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and preprocess the Iris dataset
data = load_iris()
X = data.data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Get an animation of the UMAP algorithm
umap = umap_mapping(n_neighbors=10, n_components=2, min_dist=0.1)
Y = umap.fit_transform(X, n_epochs=300, animation=True, labels = data.target)
```

By running this in the main folder, you should get an animation close to this one:

![til](umap_animation_example.gif)


## 🔧 Installation

1. Clone the repository:
```
git clone https://github.com/paquitopg/umap-clustering.git
cd umap-clustering
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Start exploring notebooks in `data_exploration/` or run scripts in the main folders.


## 📁 Datasets (Planned)

We plan to demonstrate UMAP clustering on the following datasets:

* Socio-economic and health description of countries (~9 dimensions)

* NYC Yellow Taxi Trip Data (~18 dimensions)

* French cities socio-economic indicators (~54 dimensions)

Each dataset showcases challenges like varying dimensions and features.


## 📚 References that helped us build our algorithms

1. McInnes, Leland; Healy, John; UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

2. Wei, Dong; Charikar, Moses; Kai, Li; Efficient k-nearest neighbor graph constructionfor generic similarity measures.
