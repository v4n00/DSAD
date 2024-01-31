import numpy as np
import sklearn.preprocessing as pp

x = np.ndarray() # standardized
y = np.ndarray() # standardized

n, p = x.shape
q = y.shape[1]
Cov = np.cov(x, y, rowvar=False)
CX = np.cov(x, rowvar=False)
CY = np.cov(y, rowvar=False)
invCX = np.linalg.inv(CX)
invCY = np.linalg.inv(CY)
CXY = Cov[:p, p:]
CYX = CXY.T
h1 = invCX @ CXY
h2 = invCY @ CYX
m = min(p, q)
if p == m:
    h = h1 @ h2
    r2, a = np.linalg.eig(h)
    r = np.sqrt(r2)
    b = (h2 @ a) @ np.diag(1 / r)
else:
    h = h2 @ h1
    r2, b = np.linalg.eig(h)
    r = np.sqrt(r2)
    a = (h1 @ b) @ np.diag(1 / r)
z = pp.normalize(x @ a, axis=0)
u = pp.normalize(y @ b, axis=0)

Rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, q:]