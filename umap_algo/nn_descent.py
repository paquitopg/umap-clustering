"""Algorithme basé sur le papier 
Dong, W., Moses, C., & Li, K. (2011, March). Efficient k-nearest neighbor graph construction for generic 
similarity measures. In Proceedings of the 20th international conference on World wide web (pp. 577-586)."""

import numpy as np
import heapq 
from scipy.spatial.distance import pdist, squareform

def approx_knn_all_points(X, k, metric="euclidean"):
    """
    Calcule les k plus proches voisins approximés pour tous les points du dataset en utilisant NNDescent.

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

    distance_matrix = squareform(pdist(X, metric=metric))
    sigma = -distance_matrix

    return NNDescent(X, sigma, k)



def NNDescent(V, sigma, K, max_iter=1000):
    """Algorithme 1 de l'article
    Inputs:
        V: ndarray (N, d), the dataset
        sigma: ndarray(N, N), similarity oracle
        K: int, number of neighbors
    Outputs:
        indices: liste des + proches voisins pour chacun des points du dataset
        distances: liste des distances associées
    """
    N, dim = np.shape(V)
    B = [[] for _ in range(N)]
    for v in range(N):
        samples = np.random.choice([u for u in range(N) if u != v], K, replace=False)
        for u in samples:
            heapq.heappush(B[v], (-np.inf, u))

        
    for _ in range(max_iter):
        R = reverse(B)
        B_bar = [[] for _ in range(N)]
        for v in range(N):
            for l, u in B[v]:
                heapq.heappush(B_bar[v], (l, u))
            for l, u in R[v]:
                heapq.heappush(B_bar[v], (l, u))

        c = 0

        for v in range(N):
            for _, u1 in B_bar[v]:
                for _, u2 in B_bar[u1]:
                    if u2 == v:
                        continue # Ne pas ajouter v à la liste de ses voisins
                    l = sigma[v, u2]
                    c += update_nn(B[v], (l, u2))

        if c == 0:
            break
    
    graph = [sorted([(u, -l) for l, u in heap ], key=lambda x: x[1]) for heap in B]


    indices = [[] for _ in range(N)]
    distances = [[] for _ in range(N)]
    for v in range(N) :
        for u, d in graph[v]:
            indices[v].append(u)
            distances[v].append(d)

    return indices, distances


def reverse(B):
    N = len(B)
    R = [[] for _ in range(N)]
    for u in range(N):
        for l, v in B[u]:
            R[v].append((l, u))
    return R

def update_nn(H, x):
    l, u = x

    # Si u est déjà dans le tas
    for _, v in H:
        if v == u:
            return 0
    
    # Sinon, si il a une similarité assez grande, on l'ajoute
    if H[0][0] < l:
        heapq.heapreplace(H, x)
        return 1

    return 0