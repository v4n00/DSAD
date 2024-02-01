import numpy as np
from sklearn.cross_decomposition import CCA

x = np.ndarray() # standardized
y = np.ndarray() # standardized

p = x.shape[1]
q = y.shape[1]
m = min(p, q)
modelCCA = CCA(n_components=m)
modelCCA.fit(x, y)
z, u = modelCCA.transform(x, y)
Rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, q:]