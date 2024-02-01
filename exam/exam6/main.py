import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

# ᗜˬᗜ - subiect examen furtuna 2023
rawMort = pd.read_csv('./dataIN/Mortalitate.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)

merged = rawCoduri.merge(rawMort, left_index=True, right_index=True)

# A1
merged[merged['RS'] < 0] \
.to_csv('./dataOUT/cerinta1.csv')

# A2
merged \
.groupby('Continent') \
.mean() \
.to_csv('./dataOUT/cerinta2.csv')

# B1
x = rawMort.values
means = np.mean(x, axis=0)
stds = np.std(x, axis=0)
x = (x - means) / stds

cov = np.cov(x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]

print(alpha)

# B2
a = eigenvectors[:, k_desc]
for j in range(a.shape[1]):
    if np.abs(np.min(a[:, j])) > np.abs(np.max(a[:, j])):
        a[:, j] = -a[:, j]
C = x @ a
scores = C / np.sqrt(alpha)

pd.DataFrame(data=scores, index=rawMort.index.values, columns=['C' + str(i + 1) for i in range(C.shape[1])]) \
.to_csv('./dataOUT/scoruri.csv')

# B3
plt.figure(figsize=(12, 12))
plt.title('Scoruri')
sb.heatmap(np.round(scores, 2), vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()