import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis
import sklearn.cluster as skl

# ᗜˬᗜ - subiect examen furtuna 2024
rawAlcohol = pd.read_csv('./dataIN/alcohol.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labAni = list(rawAlcohol.columns[1:].values)

merged = rawAlcohol.merge(rawCoduri, left_index=True, right_index=True) \
.drop(columns='Code')[['Continent'] + labAni]
merged.fillna(np.mean(merged[labAni], axis=0), inplace=True)

# A1
merged \
.apply(lambda row: np.average(row[labAni]), axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged[['Continent'] + labAni] \
.groupby('Continent') \
.mean() \
.idxmax(axis=1) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = merged[labAni].values
means = np.mean(x, axis=0)
stds = np.std(x, axis=0)
x = (x - means) / stds

methods = list(hic._LINKAGE_METHODS)
distances = dis._METRICS_NAMES

HC = hic.linkage(x, method='ward', metric=distances[7])
print(HC)

# B2
def clusters(h: np.ndarray, k):
    cat = hic.fcluster(h, k, criterion='maxclust')
    return ['C' + str(i) for i in cat]

labels = clusters(HC, 5)
merged['Clusters'] = labels
merged[['Clusters']].to_csv('./dataOUT/p4.csv')

# B3
cov = np.cov(x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
a = eigenvectors[:, k_desc]
for j in range(a.shape[1]):
    if np.abs(np.min(a[:, j])) > np.abs(np.max(a[:, j])):
        a[:, j] = -a[:, j]
C = x @ a

kmeans = skl.KMeans(n_clusters=5, n_init=10)
kmeans_labels = kmeans.fit_predict(C)
plt.figure(figsize=(8, 6))
plt.scatter(C[:, 0], C[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-means Clustering on PCA Data")
plt.show()

# C
# nu stiu lol ᗜˬᗜ