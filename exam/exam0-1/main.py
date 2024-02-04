import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2024 Vinte
# dataset is generated with ChatGPT
rawNatLoc = pd.read_csv('./dataIN/NatLocMovements.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/PopulationLoc.csv', index_col=0)
labels = list(rawNatLoc.columns.values[1:])

merged = rawNatLoc.merge(rawPop, right_index=True, left_index=True)[['City', 'CountyCode', 'Population'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged \
.groupby('CountyCode') \
.sum() \
.apply(lambda row: (row['LiveBirths'] / (row['Population'] / 1000)) - (row['Deceased'] / (row['Population'] / 1000)), axis=1) \
.to_csv('./dataOUT/Request_1.csv')

# A2
merged \
.set_index(['City', 'CountyCode']) \
.apply(lambda row: row[labels] / (row['Population'] / 1000), axis=1) \
.reset_index(1) \
.groupby('CountyCode') \
.apply(lambda df: pd.Series({lab: df[lab].idxmax() for lab in labels})) \
.to_csv('./dataOUT/Request_2.csv')

# B1
rawHealth = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)

x = StandardScaler().fit_transform(rawHealth)
pd.DataFrame(x, columns=rawHealth.columns.values).to_csv('./dataOUT/Xstd.csv')

HC = linkage(x, method='ward')
print(HC)

# B2
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

print('junction with max df:', j)
print('threshold:', np.round(t, 2))

# B3
plt.figure(figsize=(12, 12))
plt.title('Dendogram')
dendrogram(HC, labels=rawHealth.index.values, leaf_rotation=45)
plt.axhline(t, c='r')
plt.show()