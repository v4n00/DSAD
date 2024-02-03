import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2024 Furtuna
# dataset is generated with ChatGPT
rawEmmi = pd.read_csv('./dataIN/emmissions.csv', index_col=0)
rawCodes = pd.read_csv('./dataIN/PopulatieEuropa.csv', index_col=0)
labels = list(rawEmmi.columns[1:].values)

merged = rawCodes.merge(rawEmmi[labels], left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged[labels[:-2]] = merged[labels[:-2]] * 1000
merged[['Country'] + labels] \
.set_index('Country', append=True) \
.sum(axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged[['Region', 'Population'] + labels] \
.groupby('Region') \
.sum() \
.apply(lambda row: row[labels] / (row['Population'] / 100_000), axis=1) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
# I don't want to generate another dataset, so I just split the original one
labels1 = labels[:4]
labels2 = labels[4:]
x = StandardScaler().fit_transform(merged[labels1])
y = StandardScaler().fit_transform(merged[labels2])

p = x.shape[1]
q = y.shape[1]
m = min(p, q)
cca = CCA(n_components=m)
z, u = cca.fit_transform(x, y)

zlabels = ['Z' + str(i + 1) for i in range(z.shape[1])]
ulabels = ['U' + str(i + 1) for i in range(u.shape[1])]
pd.DataFrame(z, index=merged.index.values, columns=zlabels).to_csv('./dataOUT/z.csv')
pd.DataFrame(u, index=merged.index.values, columns=ulabels).to_csv('./dataOUT/u.csv')

# B2
r = []
for i in range(m):
    r.append(np.corrcoef(z[:, i], u[:, i], rowvar=False)[0, 1])

pd.DataFrame(r).to_csv('./dataOUT/r.csv')

# B3
# I won't do bartlett

# C
# I don't know