import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2023 Furtuna
rawMort = pd.read_csv('./dataIN/Mortalitate.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labels = rawMort.columns.values

merged = rawCoduri.merge(rawMort, left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged[merged['RS'] < 0] \
.to_csv('./dataOUT/cerinta1.csv')

# A2
merged \
.groupby('Continent') \
.mean() \
.to_csv('./dataOUT/cerinta2.csv')

# B1
x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_

print(alpha)

# B2
scores = C / np.sqrt(alpha)

pd.DataFrame(data=scores, index=rawMort.index.values, columns=['C' + str(i + 1) for i in range(C.shape[1])]) \
.to_csv('./dataOUT/scoruri.csv')

# B3
plt.figure(figsize=(12, 12))
plt.title('Scoruri')
plt.scatter(scores[:, 0], scores[:, 1])
plt.show()