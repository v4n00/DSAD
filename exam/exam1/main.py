import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as skl

rawInd = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)

actLab = list(rawInd.columns.values[1:])
merged = rawInd.merge(right=rawPop, right_index=True, left_index=True)

# req 1
c1 = merged.copy()
for act in actLab:
    c1[act] = c1[act] / c1['Populatie']
c1[['Localitate_x'] + actLab].to_csv('./dataOUT/Request_1.csv')

print(merged)
# req 2
c2 = merged.copy()
c2 = c2[['Judet'] + actLab].groupby('Judet').sum()
c2['Cifra de afaceri'] = c2.max(axis=1)
c2['Industrie'] = c2.idxmax(axis=1)
c2[['Industrie', 'Cifra de afaceri']].to_csv('./dataOUT/Request_2.csv')

# req 3
def standardise(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds

rawProd = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
indexes = rawProd.index
prodLab = rawProd.columns[:4]
conLab = rawProd.columns[4:]

X = pd.DataFrame(data=rawProd[prodLab].values, index=indexes, columns=prodLab)
Y = pd.DataFrame(data=rawProd[conLab].values, index=indexes, columns=conLab)
Xstd = pd.DataFrame(data=standardise(X.values), index=indexes, columns=prodLab)
Ystd = pd.DataFrame(data=standardise(Y.values), index=indexes, columns=conLab)

Xstd.to_csv('./dataOUT/Xstd.csv')
Ystd.to_csv('./dataOUT/Ystd.csv')

# req 4
n, p = Xstd.shape
q = Ystd.shape[1]
m = min(p, q)
modelCCA = skl.CCA(n_components=m)
modelCCA.fit(Xstd, Ystd)
z, u = modelCCA.transform(Xstd, Ystd)

ZLab = ['Z' + str(i+1) for i in range(z.shape[1])]
ULab = ['U' + str(i+1) for i in range(u.shape[1])]
pd.DataFrame(data=z, index=indexes, columns=ZLab).to_csv('./dataOUT/Xscore.csv')
pd.DataFrame(data=u, index=indexes, columns=ULab).to_csv('./dataOUT/Yscore.csv')

# req 5
Rxz = modelCCA.x_loadings_
Ryu = modelCCA.y_loadings_

pd.DataFrame(data=Rxz, index=ZLab, columns=prodLab).to_csv('./dataOUT/Rxz.csv')
pd.DataFrame(data=Ryu, index=ULab, columns=conLab).to_csv('./dataOUT/Ryu.csv')

# req 6
def biplot(x: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(7, 7))
    plt.title("Biplot (z1, u1) / (z2, u2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x[:, 0], x[:, 1], c='r', label='X')
    plt.scatter(y[:, 0], y[:, 1], c='b', label='Y')
    plt.legend()
    plt.show()

biplot(z[:, [0, 1]], u[:, [0, 1]])