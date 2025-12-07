# umap-clustering
Uniform Manifold Approximation and Projection algorithm applied to dimensionality reduction and clustering. 

The Uniform Manifold Approximation and Projection (UMAP) algorithm has been introduced by McInnes et al. in 2018 [1] as a nonlinear dimensionality reduction (DR) technique. It builds upon concepts from manifold theory and topological data analysis to produce meaningful low-dimensional representations of high-dimensional data. Compared to earlier methods such as t-SNE, UMAP offers better preservation of global structure, and scalability to large datasets. Beyond visualization,  UMAP has been explored as a tool for tasks such as clustering [2]. 
For this project, we plan to explore the UMAP algorithm and its application to real datasets. We want to compare its performance to other previous DR techniques such as PCA or the more advanced t-SNE.  


Datasets

We plan to use 3 datasets that have different dimensions but are all based on geographical data:
Socio-economic and health description of countries [3]: 9 dimensions, geographical coordinates may be added if needed.
NYC Yellow Taxi Trip Data [4]: 18 dimensions
100 French cities indicators (economics, health, risk factors, nature, culture) [5]: 54 columns. Using geographical coordinates for the whole country or regions is possible thanks to French public data [6].

We would use them this way:
Apply UMAP (vs. other dimensionality reduction algorithms) on the data.
Run basic clustering algorithms (e.g., KNN, DBSCAN) with different parameters.
Use silhouette score (or other clustering metrics if needed) to compare UMAP vs. other dimensionality reduction algorithms.


References

[1] McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018

[2] Delahoz-Domínguez, E., Mendoza-Mendoza, A., & Visbal-Cadavid, D. (2025). Clustering of Countries Through UMAP and K-Means: A Multidimensional Analysis of Development, Governance, and Logistics. Logistics, 9(3), 108.

[3] Unsupervised Learning on Country Data, Kaggle
 https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data 

[4] NYC Yellow Taxi Trip Data, Kaggle
https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data

[5] French cities, Kaggle
https://www.kaggle.com/datasets/zakariakhodri3/french-cities-2005

[6] Datagouv geographical data
https://www.data.gouv.fr/pages/donnees-geographiques 
