import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2024 Furtuna
rawAlcohol = pd.read_csv('./dataIN/alcohol.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labels = list(rawAlcohol.columns[1:].values)

merged = rawAlcohol.merge(rawCoduri, left_index=True, right_index=True) \
.drop(columns='Code')[['Continent'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged \
.apply(lambda row: np.average(row[labels]), axis=1) \
.sort_values(ascending=False) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged[['Continent'] + labels] \
.groupby('Continent') \
.mean() \
.idxmax(axis=1) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = StandardScaler().fit_transform(merged[labels])

HC = linkage(x, method='ward')
print(HC)

# B2
cat = fcluster(HC, 5, criterion='maxclust')
clusters =  ['C' + str(i) for i in cat]

merged['Clusters'] = clusters
merged[['Clusters']].to_csv('./dataOUT/p4.csv')

# B3
pca = PCA()
C = pca.fit_transform(x)

kmeans = KMeans(n_clusters=5, n_init=10)
kmeans_labels = kmeans.fit_predict(C)
plt.figure(figsize=(8, 6))
plt.scatter(C[:, 0], C[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-means Clustering on PCA Data")
plt.show()

# C
# I don't know