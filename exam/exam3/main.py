import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis

# ᗜˬᗜ - subiect examen furtuna 2024
rawAir = pd.read_csv('./dataIN/AirQuality.csv', index_col=0)
rawContinent = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labAir = list(rawAir.columns.values[1:])

merged = rawAir.merge(rawContinent, left_index=True, right_index=True) \
.drop('Country_y', axis=1).rename(columns={'Country_x': 'Country'})[['Continent', 'Country'] + labAir]

# A1
merged[['Country'] + labAir] \
.set_index('Country') \
.idxmax(axis=0) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged.set_index('Country') \
.groupby(['Continent']) \
.apply(func=lambda df: pd.Series({ind: df[ind].idxmax() for ind in labAir})) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
methods = list(hic._LINKAGE_METHODS)
distances = dis._METRICS_NAMES

HC = hic.linkage(rawAir[labAir].values, method='ward', metric=distances[7])
print(HC)

# B2
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

plt.figure(figsize=(12, 7))
plt.title('Dendogram')
hic.dendrogram(HC, leaf_rotation=30, labels=merged['Country'].values)
plt.axhline(t, c='r')
plt.show()

# B3
def clusters(h: np.ndarray, k):
    n = h.shape[0] + 1
    g =  np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    cat = pd.Categorical(g)
    return ['C' + str(i + 1) for i in cat.codes]

merged['Cluster'] = clusters(HC, n - j)
merged[['Country', 'Cluster']].to_csv('./dataOUT/popt.csv')

# C
# nu stiu lol ᗜˬᗜ