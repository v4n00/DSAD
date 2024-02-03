import numpy as np
from sklearn.cross_decomposition import CCA

x = np.ndarray() # standardized
y = np.ndarray() # standardized

p = x.shape[1]
q = y.shape[1]
m = min(p, q)
cca = CCA(n_components=m)
z, u = cca.fit_transform(x, y)
rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, q:]
r = []
for i in range(m):
    r.append(np.corrcoef(z[:, i], u[:, i], rowvar=False)[0, 1])