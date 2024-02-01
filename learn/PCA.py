import numpy as np
from sklearn.decomposition import PCA

x = np.ndarray() # standardized

pca = PCA()
pca.fit(x)
alpha = pca.explained_variance_
a = pca.components_
C = pca.transform(x)
rxc = a * np.sqrt(alpha) # factor loadings
scores = C / np.sqrt(alpha)
C2 = C * C
quality = np.transpose(C2.T / np.sum(C2, axis=1))
contributions = C2 / (x.shape[0] * alpha)
communalities = np.cumsum(rxc * rxc, axis=1) # correlation between observed variables and principal components
pve = pca.explained_variance_ratio_