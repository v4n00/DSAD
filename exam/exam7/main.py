import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis

# ᗜˬᗜ - subiect examen furtuna 2023
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
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

plt.figure(figsize=(12, 12))
plt.title('Dendogram')
hic.dendrogram(HC, leaf_rotation=30, labels=merged.index.values)
plt.axhline(t, c='r')
plt.show()

# B3
def clusters(h: np.ndarray, k):
    cat = hic.fcluster(h, k, criterion='maxclust')
    return ['C' + str(i) for i in cat]

labels = clusters(HC, n - j)
merged['Clusters'] = labels
merged['Clusters'].to_csv('./dataOUT/popt.csv')
