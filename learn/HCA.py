import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis
from scipy.cluster.hierarchy import _LINKAGE_METHODS, fcluster, linkage
from scipy.spatial.distance import _METRICS_NAMES
from sklearn.cluster import KMeans

x = np.ndarray() # standardized

def threshold(h: np.ndarray):
    n = h.shape[0] # no. of junctions
    dist_1 = h[1:n, 2]
    dist_2 = h[0:n - 1, 2]
    diff = dist_1 - dist_2
    j = np.argmax(diff) # junction with max. diff
    t = (h[j, 2] + h[j + 1, 2]) / 2 # threshold
    return t, j, n

def clusters(h: np.ndarray, k):
    cat = fcluster(h, k, criterion='maxclust')
    return ['C' + str(i) for i in cat]

methods = list(_LINKAGE_METHODS)
distances = _METRICS_NAMES

# methods[] ultimele 4
# folosesc distances[7]
# restul folosesc distances[3]
HC = linkage(x, method=methods[3], metric=distances[7])
t, j, n = threshold(HC)
k = n - j

# determine the clusters belonging to the maximum stability partition
labels = clusters(HC, k) # add this to the original DataFrame

# KMeans
C = np.ndarray() # principal components
noClusters = 5 # how many clusters you want
kmeans = KMeans(n_clusters=noClusters, n_init=10)
kmeans_labels = kmeans.fit_predict(C)
plt.scatter(C[:, 0], C[:, 1], c=kmeans_labels, cmap='viridis')