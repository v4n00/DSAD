import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subiect examen furtuna 2024
rawAir = pd.read_csv('./dataIN/AirQuality.csv', index_col=0)
rawContinent = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labels = list(rawAir.columns.values[1:])

merged = rawAir.merge(rawContinent, left_index=True, right_index=True) \
.drop('Country_y', axis=1).rename(columns={'Country_x': 'Country'})[['Continent', 'Country'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged[['Country'] + labels] \
.set_index('Country') \
.idxmax(axis=0) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged.set_index('Country') \
.groupby(['Continent']) \
.apply(func=lambda df: pd.Series({ind: df[ind].idxmax() for ind in labels})) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = StandardScaler().fit_transform(rawAir[labels])
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

plt.figure(figsize=(12, 7))
plt.title('Dendogram')
hic.dendrogram(HC, leaf_rotation=30, labels=merged['Country'].values)
plt.axhline(t, c='r')
plt.show()

# B3
cat = hic.fcluster(HC, n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]

merged['Cluster'] = clusters
merged[['Country', 'Cluster']].to_csv('./dataOUT/popt.csv')

# C
# nu stiu lol ᗜˬᗜ