# Data Structure
import pandas as pd
import numpy as np

# Plot
import matplotlib.pyplot as plt

# Data processing
from sklearn.decomposition import PCA
from knn import exact_knn_all_points

class umap_mapping:
    def __init__(self, n_neighbors=15, min_dist=0.1, metric='euclidean', KNN_method='exact', random_state=42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.KNN_method = KNN_method
        self.random_state = random_state

    def KNN_graph(self, X): 
        K = self.n_neighbors

        if self.KNN_method == 'exact':
            indices, distances = exact_knn_all_points(X, k=K, metric=self.metric)
            return [[indices[i], distances[i]] for i in range(len(X))]
        
    def rho(self,KNN_graph):
        return [min([d for d in KNN_graph[i][1] if d > 0]) for i in range(len(KNN_graph))]
        

    def plot(self, clusters=None):
        plt.figure(figsize=(7, 5))
        plt.scatter(self.X_umap[:, 0], self.X_umap[:, 1], c=clusters, cmap="tab10")
        plt.title("UMAP")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.show()
