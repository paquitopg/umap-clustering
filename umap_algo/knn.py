import numpy as np
from sklearn.neighbors import KDTree

def exact_knn_all_points(X, k, metric="euclidean"):
    """
    Compute the k nearest neighbors for all points in dataset X using KDTree.

    Parameters
    ----------
    X : ndarray (N, d) - dataset
    k : int - number of neighbors
    metric : str - distance (euclidean, manhattan, etc.)

    Returns
    -------
    indices : ndarray (N, k) - indices of the k nearest neighbors
    distances : ndarray (N, k) - associated distances
    """

    tree = KDTree(X, metric=metric)

    # k+1 car le point lui-même est retourné
    distances, indices = tree.query(X, k=k + 1)

    # On enlève le point lui-même (distance nulle)
    return indices[:, 1:], distances[:, 1:]
