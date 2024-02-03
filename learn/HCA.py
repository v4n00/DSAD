import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

x = np.ndarray() # standardized

def threshold(h: np.ndarray):
    n = h.shape[0] # no of junctions
    dist_1 = h[1:n, 2]
    dist_2 = h[0:n - 1, 2]
    diff = dist_1 - dist_2
    j = np.argmax(diff) # junction with max. diff
    t = (h[j, 2] + h[j + 1, 2]) / 2 # threshold
    return t, j, n

def clusters(h: np.ndarray, k):
    cat = fcluster(h, k, criterion='maxclust')
    return ['C' + str(i) for i in cat]

HC = linkage(x, method='ward') # the method is given in the requirements
t, j, n = threshold(HC)

# determine the clusters belonging to the maximum stability partition
k = n - j
labels = clusters(HC, k) # add this to the original DataFrame

# KMeans
C = np.ndarray() # principal components
kmeans = KMeans(n_clusters=5, n_init=10) # n_clusters is given in the requirements
kmeans_labels = kmeans.fit_predict(C)
plt.scatter(C[:, 0], C[:, 1], c=kmeans_labels, cmap='viridis')