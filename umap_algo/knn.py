import numpy as np
from sklearn.neighbors import KDTree

def exact_knn_all_points(X, k, metric="euclidean"):
    """
    Calcule les k plus proches voisins exacts pour tous les points du dataset.

    Parameters
    ----------
    X : ndarray (N, d)
        Dataset
    k : int
        Nombre de voisins
    metric : str
        Distance (euclidean, manhattan, etc.)

    Returns
    -------
    indices : ndarray (N, k)
        Indices des k plus proches voisins
    distances : ndarray (N, k)
        Distances associées
    """

    tree = KDTree(X, metric=metric)

    # k+1 car le point lui-même est retourné
    distances, indices = tree.query(X, k=k + 1)

    # On enlève le point lui-même (distance nulle)
    return indices[:, 1:], distances[:, 1:]
