import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis

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
    n = h.shape[0] + 1
    g =  np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    cat = pd.Categorical(g)
    return ['C' + str(i) for i in cat.codes], cat.codes

methods = list(hic._LINKAGE_METHODS)
distances = dis._METRICS_NAMES

# methods[] ultimele 4
# folosesc distances[7]
# restul folosesc distances[3]
HC = hic.linkage(x, method=methods[3], metric=distances[7])
t, j, n = threshold(HC)
k = n - j

# determine the clusters belonging to the maximum stability partition
labels, codes = clusters(HC, k)
print(labels)
print(codes)