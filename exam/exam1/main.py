import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject tutoring 2023/2024 Vinte
rawInd = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)
labels = list(rawInd.columns.values[1:])

merged = rawInd.merge(right=rawPop, right_index=True, left_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# req 1
merged \
.apply(lambda row: row[labels] / row['Populatie'], axis=1) \
.to_csv('./dataOUT/Request_1.csv')

# req 2
c2 = merged[['Judet'] + labels].groupby('Judet').sum()
c2['Cifra de afaceri'] = c2.max(axis=1)
c2['Industrie'] = c2.idxmax(axis=1)
c2[['Industrie', 'Cifra de afaceri']] \
.to_csv('./dataOUT/Request_2.csv')

# req 3
rawProd = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
indexes = rawProd.index
prodLab = rawProd.columns[:4]
conLab = rawProd.columns[4:]

x = pd.DataFrame(data=StandardScaler().fit_transform(rawProd[prodLab]), index=indexes, columns=prodLab)
y = pd.DataFrame(data=StandardScaler().fit_transform(rawProd[conLab]), index=indexes, columns=conLab)

x.to_csv('./dataOUT/Xstd.csv')
y.to_csv('./dataOUT/Ystd.csv')

# req 4
n, p = x.shape
q = y.shape[1]
m = min(p, q)
modelCCA = CCA(n_components=m)
modelCCA.fit(x, y)
z, u = modelCCA.transform(x, y)

ZLab = ['Z' + str(i+1) for i in range(z.shape[1])]
ULab = ['U' + str(i+1) for i in range(u.shape[1])]
pd.DataFrame(data=z, index=indexes, columns=ZLab).to_csv('./dataOUT/Xscore.csv')
pd.DataFrame(data=u, index=indexes, columns=ULab).to_csv('./dataOUT/Yscore.csv')

# req 5
Rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, q:]

pd.DataFrame(data=Rxz, index=ZLab, columns=prodLab).to_csv('./dataOUT/Rxz.csv')
pd.DataFrame(data=Ryu, index=ULab, columns=conLab).to_csv('./dataOUT/Ryu.csv')

# req 6
plt.figure(figsize=(7, 7))
plt.title("Biplot (z1, u1) / (z2, u2)")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(z[:, 0], z[:, 1], c='r', label='X')
plt.scatter(u[:, 0], u[:, 1], c='b', label='Y')
plt.legend()
plt.show()