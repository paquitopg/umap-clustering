# Data Structure
import pandas as pd
import numpy as np

# Plot
import matplotlib.pyplot as plt

# Utils
from sklearn.decomposition import PCA
from knn import exact_knn_all_points
from scipy.optimize import root_scalar, curve_fit
from sklearn.preprocessing import StandardScaler

class umap_mapping:
    def __init__(self, n_neighbors=15, n_components=2, min_dist=0.1, KNN_metric = 'euclidean', KNN_method='exact', random_state=42):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = KNN_metric
        self.KNN_method = KNN_method
        self.random_state = random_state

        # Taking default values for a and b, replaced later by fitting
        self.a = 1.9
        self.b = 0.79

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
        rho: array-like, shape (n_samples,) - the distance to the closest neighbor (non-zero) for each point
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
        rho: array-like, shape (n_samples,) - the distance to the closest neighbor (non-zero) for each point
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


    def attractive_force(self, y_i, y_j, weight_ij): # See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (-2*self.a*self.b*np.linalg.norm(y_i - y_j)**(2 * self.b - 2))/(1 + np.linalg.norm(y_i - y_j) ** 2) * (y_i - y_j) * weight_ij

    
    def repulsive_force(self, y_i, y_j, weight_ij, epsilon=1e-3): #See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (2*self.b) / ( (epsilon + np.linalg.norm(y_i - y_j)**2)*(1 + self.a*np.linalg.norm(y_i - y_j)**(2*self.b))) * (1-weight_ij) * (y_i - y_j)
    
    def find_ab_params(self, KNN_graph):
        """
        Fit the parameters a and b for the UMAP attractive and repulsive forces by 
        non-linear least squares fitting against the curve.
        (see Definition 11 and equation (17) of appendix C of the UMAP paper https://arxiv.org/pdf/1802.03426)

        ---------
        Inputs:
        KNN_graph: adjacency list of length n_samples, each element is a tuple (index in X, distance)

        Returns:
        a, b: float - parameters for the attractive and repulsive forces
        --------- 
        """
        def curve(d, a, b):
            return 1 / (1 + a * d ** (2 * b))

        d = np.array([x[1] for x in KNN_graph]).flatten()  # distances to neighbors
        psi = np.where(d <= self.min_dist, 1.0, np.exp(-(d - self.min_dist)))
        
        (a, b), _ = curve_fit(curve, d, psi)

        return a, b


    def PCA_embedding(self, X):
        return PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)

    def random_embedding(self, n_samples):
        np.random.seed(self.random_state)
        return np.random.randn(n_samples, self.n_components)

    def spectral_embedding(self, KNN_graph, weights):
        n_samples = len(KNN_graph)
        A = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            neighbors = KNN_graph[i][0]
            for idx, j in enumerate(neighbors):
                A[i, j] = weights[i][idx]

        D = np.diag(A.sum(axis=1))
        L = D**0.5 @ (D - A) @ D**0.5

        eigvals, eigvecs = np.linalg.eigh(L)

        return eigvecs[:, 1:self.n_components+1]


    def optimize(self, Y, KNN_graph, weights, n_epochs=200, learning_rate=1):
        """
        Optimize the low-dimensional embedding Y using stochastic gradient descent.

        ---------
        Inputs:
        Y: array-like, shape (n_samples, n_components) - initial embedding
        KNN_graph: adjacency list of length n_samples, each element is a tuple (indexes in X, distances)
        weights: array-like, shape (n_samples, n_neighbors) - adjusted weights
        a, b: float - parameters for the attractive and repulsive forces
        n_epochs: int - number of epochs for optimization
        learning_rate: float - initial learning rate for optimization

        Returns:
        Y: array-like, shape (n_samples, n_components) - optimized embedding
        ---------
        """

        n_samples = Y.shape[0]
        n_neg = 5

        for epoch in range(n_epochs):
            for i in range(n_samples):
                for idx, j in enumerate(KNN_graph[i][0]):
                    if i == j:
                        continue

                    w_ij = weights[i][idx]

                    if np.random.rand() > w_ij:
                        continue

                    grad_attr = self.attractive_force(Y[i], Y[j], w_ij)

                    Y[i] += learning_rate * grad_attr

                for _ in range(n_neg):
                    k = np.random.randint(0, n_samples)
                    if k == i:
                        continue
                    
                    # Search if k is in the neighbors of i
                    w_ik = 0.0
                    if k in KNN_graph[i][0]:
                        idx_k = np.where(KNN_graph[i][0] == k)[0][0]
                        w_ik = weights[i][idx_k]

                    grad_rep = self.repulsive_force(Y[i], Y[k], w_ik)
                    Y[i] += learning_rate * grad_rep


            learning_rate -= epoch / n_epochs * learning_rate

        return Y

    def fit_transform(self, X, n_epochs=200):
        """
        Fit the UMAP model to the data X and transform it into a low-dimensional embedding.

        ---------
        Inputs:
        X: array-like, shape (n_samples, n_features)
        n_epochs: int - number of epochs for optimization

        Returns:
        Y: array-like, shape (n_samples, n_components) - low-dimensional embedding
        ---------
        """
        # 1. KNN
        KNN_graph = self.compute_KNN_graph(X)

        # 2. rho & sigma
        rho, sigma = self.rho_sigma(KNN_graph)

        # 3. poids symétrisés
        weights = self.compute_adjusted_weights(KNN_graph, rho, sigma)

        # 4. paramètres a, b
        self.a, self.b = self.find_ab_params(KNN_graph)

        # 5. init embedding
        Y = self.spectral_embedding(KNN_graph, weights)

        plt.scatter(Y[:, 0], Y[:, 1], c=data.target)
        plt.title("UMAP Embedding of Iris Dataset")
        plt.show()

        # 6. optimisation
        Y = self.optimize(Y, KNN_graph, weights, n_epochs=n_epochs)

        return Y

if __name__ == "__main__":
    # Exemple d'utilisation
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    umap = umap_mapping(n_neighbors=10, n_components=2, min_dist=0.1, random_state=42)
    Y = umap.fit_transform(X, n_epochs=500)

    plt.scatter(Y[:, 0], Y[:, 1], c=data.target)
    plt.title("UMAP Embedding of Iris Dataset")
    plt.show()
