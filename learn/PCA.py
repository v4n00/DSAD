import numpy as np

x = np.ndarray() # standardized

cov = np.cov(x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
a = eigenvectors[:, k_desc]
for j in range(a.shape[1]):
    if np.abs(np.min(a[:, j])) > np.abs(np.max(a[:, j])):
        a[:, j] = -a[:, j]
C = x @ a
rxc = a * np.sqrt(alpha)
scores = C / np.sqrt(alpha)
C2 = C * C
quality = np.transpose(C2.T / np.sum(C2, axis=1))
contributions = C2 / (x.shape[0] * alpha)
commonalities = np.cumsum(rxc * rxc, axis=1)