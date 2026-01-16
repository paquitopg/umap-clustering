# Data Structure
import numpy as np
import scipy.sparse as sp

# Plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Utils
from .knn import exact_knn_all_points
from .nn_descent import approx_knn_all_points
from scipy.optimize import root_scalar, curve_fit

class umap_mapping:
    def __init__(self, n_neighbors=15, n_components=2, min_dist=0.1, KNN_metric = 'euclidean', KNN_method='exact'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = KNN_metric
        self.KNN_method = KNN_method

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
        distance_matrix: sparse matrix, shape (n_samples, n_samples) - distance matrix of the KNN graph
        ---------
        """
        K = self.n_neighbors

        if self.KNN_method == 'exact':
            indices, distances = exact_knn_all_points(X, k=K, metric=self.metric)

        else: # approximate KNN
            indices, distances = approx_knn_all_points(X, k=K, metric=self.metric)

        # Build distance matrix

        n_samples = len(X)
        distance_matrix = sp.csr_matrix((n_samples, n_samples))

        for i in range(n_samples):
            for j in indices[i]:
                distance_matrix[i, j] = distances[i][np.where(indices[i] == j)[0][0]]

        return distance_matrix
        

    def rho_sigma(self, distance_matrix):
        """
        Compute rho and sigma for each point in the KNN graph.
        For each point i, rho_i is the distance to the closest neighbor (non-zero),
        and sigma_i is computed as in the UMAP paper (Part 3.1 https://arxiv.org/pdf/1802.03426).
        
        ---------
        Inputs:
        distance_matrix: sparse matrix, shape (n_samples, n_samples) - distance matrix of the KNN graph

        Returns:
        rho: array-like, shape (n_samples,) - the distance to the closest neighbor (non-zero) for each point
        sigma: array-like, shape (n_samples,)
        ---------
        """
        
        rho = distance_matrix.min(axis=1, explicit=True).toarray().flatten()

        def func(sigma, distances, rho):
            return sum(np.exp(-(np.maximum(0, distances - rho)) / sigma)) - np.log2(self.n_neighbors)

        sigma = np.ones(distance_matrix.shape[0])
        for i in range(distance_matrix.shape[0]):
            distances = distance_matrix[i].toarray().flatten()
            distances = distances[distances > 0]
            rho_i = rho[i]
            sol = root_scalar(func, args=(distances, rho_i), bracket=[1e-5, 1e5], method='bisect')
            sigma[i] = sol.root

        return rho, sigma


    def compute_adjusted_weights(self, distance_matrix, rho, sigma):
        """
        Compute the adjusted weights for the KNN graph using fuzzy union.
        
        ---------
        Inputs:
        distance_matrix: sparse matrix, shape (n_samples, n_samples) - distance matrix of the KNN graph
        rho: array-like, shape (n_samples,) - the distance to the closest neighbor (non-zero) for each point
        sigma: array-like, shape (n_samples,)

        Returns:
        adjusted_weights: sparse matrix, shape (n_samples, n_samples) - adjusted weights of the KNN graph
        ---------
        """

        # Directional weights
        weights = distance_matrix.copy()

        for i in range(weights.shape[0]):   #Compute the weights according to UMAP formula and keeping low memory usage
            row_slice = slice(weights.indptr[i], weights.indptr[i+1])
            weights.data[row_slice] = np.exp(-(np.maximum(0, weights.data[row_slice] - rho[i])) / sigma[i])

      
        # Symmetric weights (fuzzy union)
        return weights + weights.T - weights.multiply(weights.T)


    def attractive_force(self, y_i, y_j, weight_ij): # See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (-2*self.a*self.b*np.linalg.norm(y_i - y_j)**(2 * self.b - 2))/(1 + self.a * np.linalg.norm(y_i - y_j) ** (2*self.b)) * (y_i - y_j) * weight_ij

    
    def repulsive_force(self, y_i, y_j, weight_ij, epsilon=1e-3): #See Part 3.2 https://arxiv.org/pdf/1802.03426
        return (2*self.b) / ( (epsilon + np.linalg.norm(y_i - y_j)**2)*(1 + self.a*np.linalg.norm(y_i - y_j)**(2*self.b))) * (1-weight_ij) * (y_i - y_j)
    
    def find_ab_params(self, distance_matrix):
        """
        Fit the parameters a and b for the UMAP attractive and repulsive forces by 
        non-linear least squares fitting against the curve.
        (see Definition 11 and equation (17) of appendix C of the UMAP paper https://arxiv.org/pdf/1802.03426)

        ---------
        Inputs:
        distance_matrix: sparse matrix, shape (n_samples, n_samples) - distance matrix of the KNN graph

        Returns:
        a, b: float - parameters for the attractive and repulsive forces
        --------- 
        """
        def curve(d, a, b):
            return 1 / (1 + a * d ** (2 * b))

        d = distance_matrix.data.astype(np.float64)

        psi = np.where(d <= self.min_dist, 1.0, np.exp(-(d - self.min_dist)))
        
        (a, b), _ = curve_fit(curve, d, psi)

        return a, b

    def spectral_embedding(self, weights):

        deg = np.asarray(weights.sum(axis=1)).ravel()

        D = sp.diags(deg)
        D_inv_sqrt = sp.diags(1/np.sqrt(deg))

        L = D_inv_sqrt.dot(D - weights).dot(D_inv_sqrt)

        eigvals, eigvecs = sp.linalg.eigsh(L, k=self.n_components + 1, which='SM')

        return eigvecs[:, 1:self.n_components+1]


    def optimize(self, Y, weights, n_epochs=200, learning_rate=0.01):
        """
        Optimize the low-dimensional embedding Y using stochastic gradient descent.

        ---------
        Inputs:
        Y: array-like, shape (n_samples, n_components) - initial embedding
        weights: sparse matrix, shape (n_samples, n_samples) - adjusted weights of the KNN graph
        a, b: float - parameters for the attractive and repulsive forces
        n_epochs: int - number of epochs for optimization
        learning_rate: float - initial learning rate for optimization

        Returns:
        Y: array-like, shape (n_samples, n_components) - optimized embedding
        ---------
        """

        n_samples = Y.shape[0]
        n_neg = 5

        # For faster computations
        indptr = weights.indptr
        indices = weights.indices
        data = weights.data

        for epoch in range(n_epochs):
            for i in range(n_samples):
                yi = Y[i]

                row_start = indptr[i]
                row_end = indptr[i + 1]

                # Attractive forces
                for idx in range(row_start, row_end):
                    j = indices[idx]
                    if j == i:
                        continue

                    w_ij = data[idx]

                    if np.random.random() > w_ij:
                        continue

                    grad = self.attractive_force(yi, Y[j], w_ij)
                    yi += learning_rate * grad

                # Negative sampling
                for _ in range(n_neg):
                    k = np.random.randint(0, n_samples)
                    if k == i:
                        continue

                    w_ik = 0.0
                    if k in indices[row_start:row_end]:
                        k_idx = np.where(indices[row_start:row_end] == k)[0][0] + row_start
                        w_ik = data[k_idx]

                    grad = self.repulsive_force(yi, Y[k], w_ik)
                    yi += learning_rate * grad

                Y[i] = yi

            learning_rate -= 1 / n_epochs * learning_rate

        return Y
    
    def optimize_generator(self, Y, weights, n_epochs=200, learning_rate=0.01):
        """
        Generator version of the optimize function to create animations.
        """

        n_samples = Y.shape[0]
        n_neg = 5

        indptr = weights.indptr
        indices = weights.indices
        data = weights.data

        for epoch in range(n_epochs):
            for i in range(n_samples):
                yi = Y[i]

                row_start = indptr[i]
                row_end = indptr[i + 1]

                # Attractive forces
                for idx in range(row_start, row_end):
                    j = indices[idx]
                    if j == i:
                        continue

                    w_ij = data[idx]

                    if np.random.random() > w_ij:
                        continue

                    grad = self.attractive_force(yi, Y[j], w_ij)
                    yi += learning_rate * grad

                # Negative sampling
                for _ in range(n_neg):
                    k = np.random.randint(0, n_samples)
                    if k == i:
                        continue

                    w_ik = 0.0
                    if k in indices[row_start:row_end]:
                        k_idx = np.where(indices[row_start:row_end] == k)[0][0] + row_start
                        w_ik = data[k_idx]

                    grad = self.repulsive_force(yi, Y[k], w_ik)
                    yi += learning_rate * grad

                Y[i] = yi

            learning_rate *= (1.0 - 1.0 / n_epochs)

            yield Y, epoch

    
    def animate_optimization(self, Y_init, weights, labels=None,n_epochs=200, learning_rate=0.01):
        """
        Only for 2D embeddings.
        Create an animation of the optimization process.
        """

        Y = Y_init.copy()

        fig, ax = plt.subplots(figsize=(6, 6))

        # For the Iris case, need to broaden the window compared to the initial embedding limits
        x_min, x_max = -4, 4
        y_min, y_max = -4, 4

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if labels is None:
            scat = ax.scatter(Y[:, 0], Y[:, 1], s=20)
        else:
            scat = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="viridis", s=20)

        ax.set_title("UMAP optimization - epoch 0")

        def update(frame):
            Y_current, epoch = frame
            scat.set_offsets(Y_current)
            ax.set_title(f"UMAP optimization - epoch {epoch}")
            return scat,

        generator = self.optimize_generator(
            Y, weights,
            n_epochs=n_epochs,
            learning_rate=learning_rate
        )

        anim = FuncAnimation(
            fig,
            update,
            frames=generator,
            interval=100,
            blit=False,
            repeat=False
        )

        plt.show()

        return anim

    def fit_transform(self, X, n_epochs=200, animation=False, labels=None, show_spectral_embedding=False, show_final_embedding=False):
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
        distance_matrix = self.compute_KNN_graph(X)

        # 2. rho & sigma
        rho, sigma = self.rho_sigma(distance_matrix)

        # 3. poids symétrisés
        weights = self.compute_adjusted_weights(distance_matrix, rho, sigma)

        # 4. a and b
        self.a, self.b = self.find_ab_params(distance_matrix)

        # 5. init embedding
        Y = self.spectral_embedding(weights)

        if self.n_components == 2 and show_spectral_embedding:
            plt.scatter(Y[:, 0], Y[:, 1], c=labels)
            plt.title("Spectral Embedding of the Dataset")
            plt.show()

        # 6. optimisation
        if animation and self.n_components == 2:
            self.animate_optimization(Y, weights, n_epochs=n_epochs, labels=labels)
        else:
            Y = self.optimize(Y, weights, n_epochs=n_epochs)

            if self.n_components == 2 and show_final_embedding:
                plt.scatter(Y[:, 0], Y[:, 1], c=labels)
                plt.title("UMAP Embedding of the Dataset")
                plt.show()

        return Y
