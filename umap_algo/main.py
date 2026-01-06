# Data Structure
import pandas as pd
import numpy as np

# Plot
import matplotlib.pyplot as plt

# Data processing
from sklearn.decomposition import PCA
from knn import exact_knn_all_points

# Utils
from scipy.optimize import root_scalar, curve_fit

class umap_mapping:
    def __init__(self, n_neighbors=15, min_dist=0.1, metric='euclidean', KNN_method='exact', random_state=42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.KNN_method = KNN_method
        self.random_state = random_state

    def compute_KNN_graph(self, X): 
        """
        Create a KNN graph from data X
        
        ---------
        Inputs:
        X: array-like, shape (n_samples, n_features)
        
        Returns:
        KNN_graph: adjacency list of length n_samples, each element is a tuple (index in X, distance)
        ---------
        """
        K = self.n_neighbors

        if self.KNN_method == 'exact':
            indices, distances = exact_knn_all_points(X, k=K, metric=self.metric)
            return [[indices[i], distances[i]] for i in range(len(X))]
        

    def rho_sigma(self, KNN_graph):
        """
        Compute rho and sigma for each point in the KNN graph.
        For each point i, rho_i is the distance to the closest neighbor (non-zero),
        and sigma_i is computed as in the UMAP paper (Part 3.1 https://arxiv.org/pdf/1802.03426).
        
        ---------
        Inputs:
        KNN_graph: adjacency list of length n_samples, each element is a tuple (index in X, distance)
        Returns:
        rho: array-like, shape (n_samples,) is the distance to the closest neighbor (non-zero) for each point
        sigma: array-like, shape (n_samples,)
        ---------
        """
        rho = np.array([min([d for d in KNN_graph[i][1] if d > 0]) for i in range(len(KNN_graph))])
        def func(sigma, distances, rho):
            return sum(np.exp(-(np.maximum(0, distances - rho)) / sigma)) - np.log2(self.n_neighbors)

        sigma = np.ones(len(KNN_graph))
        for i in range(len(KNN_graph)):
            distances = KNN_graph[i][1]
            rho_i = rho[i]
            sol = root_scalar(func, args=(distances, rho_i), bracket=[1e-5, 1e5], method='bisect')
            sigma[i] = sol.root

        return rho, sigma


    def compute_adjusted_weights(self, KNN_graph, rho, sigma):
        """
        Compute the adjusted weights for the KNN graph using fuzzy union.
        
        ---------
        Inputs:
        KNN_graph: adjacency list of length n_samples, each element is a tuple (index in X, distance)
        rho: array-like, shape (n_samples,) is the distance to the closest neighbor (non-zero) for each point
        sigma: array-like, shape (n_samples,)

        Returns:
        adjusted_weights: array-like, shape (n_samples, n_neighbors)
        ---------
        """
        n = len(KNN_graph)

        # Directional weights
        weights = []
        for i in range(n):
            distances = np.array(KNN_graph[i][1])
            w = np.exp(-(np.maximum(0, distances - rho[i])) / sigma[i])
            weights.append(w)

        # Symmetric weights (fuzzy union)
        adjusted_weights = [w.copy() for w in weights]

        for i in range(n):
            neighbors_i = KNN_graph[i][0]

            for idx_i, j in enumerate(neighbors_i):
                w_ij = weights[i][idx_i]

                # Checking if i and j are mutual neighbors
                neighbors_j = KNN_graph[j][0]

                if i in neighbors_j:
                    idx_j = np.where(neighbors_j == i)[0][0]
                    w_ji = weights[j][idx_j]
                else:
                    w_ji = 0.0

                # Fuzzy union
                adjusted_weights[i][idx_i] = w_ij + w_ji - w_ij * w_ji

        return np.array(adjusted_weights)


    def attractive_force(self, a, b, y_i, y_j, weight_ij): # See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (-2*a*b*np.linalg.norm(y_i - y_j)**(2 * b - 2))/(1 + np.linalg.norm(y_i - y_j) ** 2) * (y_i - y_j) * weight_ij

    
    def repulsive_force(self, a, b, y_i, y_j, weight_ij, epsilon=1e-3): #See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (2*b) / ( (epsilon + np.linalg.norm(y_i - y_j)**2)*(1 + a*np.linalg.norm(y_i - y_j)**(2*b))) * (1-weight_ij) * (y_i - y_j)
    
    def find_ab_params(self, spread=1.0): # Need to be reviewed and fact-checked
        def curve(d, a, b):
            return 1.0 / (1.0 + a * d ** (2 * b))

        d = np.linspace(0, spread * 3, 300)
        y = np.where(d <= self.min_dist, 1.0, np.exp(-(d - self.min_dist)))

        (a, b), _ = curve_fit(curve, d, y)
        return a, b
    
    def init_embedding(self, n_samples, n_components=2):
        np.random.seed(self.random_state)
        return np.random.randn(n_samples, n_components)


    def optimize(self, Y, KNN_graph, weights, a, b, n_epochs=200, learning_rate=1.0):
        """
        Optimize the low-dimensional embedding Y using stochastic gradient descent.

        ---------
        Inputs:
        Y: array-like, shape (n_samples, n_components) - initial embedding
        KNN_graph: adjacency list of length n_samples, each element is a tuple (index in X, distance)
        weights: array-like, shape (n_samples, n_neighbors) - adjusted weights
        a, b: float - parameters for the attractive and repulsive forces
        n_epochs: int - number of epochs for optimization
        learning_rate: float - initial learning rate for optimization

        Returns:
        Y: array-like, shape (n_samples, n_components) - optimized embedding
        ---------
        """

        n_samples = Y.shape[0]

        for epoch in range(n_epochs):
            for i in range(n_samples):
                for idx, j in enumerate(KNN_graph[i][0]):
                    if i == j:
                        continue

                    w_ij = weights[i][idx]

                    # force attractive
                    grad_attr = self.attractive_force(
                        a, b, Y[i], Y[j], w_ij
                    )

                    # force répulsive
                    grad_rep = self.repulsive_force(
                        a, b, Y[i], Y[j], w_ij
                    )

                    grad = grad_attr + grad_rep

                    Y[i] += learning_rate * grad
                    Y[j] -= learning_rate * grad

            # décroissance du LR (classique)
            learning_rate *= 0.99

        return Y
    
    def fit_transform(self, X, n_components=2, n_epochs=200):
        """
        Fit the UMAP model to the data X and transform it into a low-dimensional embedding.

        ---------
        Inputs:
        X: array-like, shape (n_samples, n_features)
        n_components: int - number of dimensions for the embedding
        n_epochs: int - number of epochs for optimization

        Returns:
        Y: array-like, shape (n_samples, n_components) - low-dimensional embedding
        ---------
        """
        # 1. KNN
        KNN = self.compute_KNN_graph(X)

        # 2. rho & sigma
        rho, sigma = self.rho_sigma(KNN)

        # 3. poids symétrisés
        weights = self.compute_adjusted_weights(KNN, rho, sigma)

        # 4. paramètres a, b
        a, b = self.find_ab_params(self.min_dist)

        # 5. init embedding
        Y = self.init_embedding(len(X), n_components)

        # 6. optimisation
        Y = self.optimize(
            Y, KNN, weights, a, b,
            n_epochs=n_epochs
        )

        return Y

if __name__ == "__main__":
    # Exemple d'utilisation
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data

    umap = umap_mapping(n_neighbors=10, min_dist=0.1, random_state=42)
    Y = umap.fit_transform(X, n_components=2, n_epochs=500)

    plt.scatter(Y[:, 0], Y[:, 1], c=data.target)
    plt.title("UMAP Embedding of Iris Dataset")
    plt.show()
