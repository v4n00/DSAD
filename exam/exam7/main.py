import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2023 Furtuna
rawAlcohol = pd.read_csv('./dataIN/alcohol.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labels = list(rawAlcohol.columns[1:].values)

merged = rawAlcohol.merge(rawCoduri, left_index=True, right_index=True) \
.drop(columns='Code')[['Continent'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged \
.apply(lambda row: np.average(row[labels]), axis=1) \
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
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

plt.figure(figsize=(12, 12))
plt.title('Dendogram')
dendrogram(HC, leaf_rotation=30, labels=merged.index.values)
plt.axhline(t, c='r')
plt.show()

# B3
cat = fcluster(HC, n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]

merged['Clusters'] = clusters
merged['Clusters'].to_csv('./dataOUT/popt.csv')
