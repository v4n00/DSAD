import numpy as np

x = np.ndarray() # standardized

cov = np.cov(x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc] # variance of the principal components
a = eigenvectors[:, k_desc]
for j in range(a.shape[1]):
    if np.abs(np.min(a[:, j])) > np.abs(np.max(a[:, j])):
        a[:, j] = -a[:, j]
C = x @ a # principal components
rxc = a * np.sqrt(alpha) # factor loadings
scores = C / np.sqrt(alpha)
C2 = C * C
quality = np.transpose(C2.T / np.sum(C2, axis=1))
contributions = C2 / (x.shape[0] * alpha)
communalities = np.cumsum(rxc * rxc, axis=1) # correlation between observed variables and principal components
pve = alpha / np.sum(alpha) # percentage of variance explained
pve_cum = np.cumsum(pve) # cumulative percentage of variance explained