import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

# ᗜˬᗜ - subiect examen furtuna 2024
rawIndicators = pd.read_csv('./dataIN/GlobalIndicatorsPerCapita_2021.csv', index_col=0)
rawContinents = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labInd = list(rawIndicators.columns.values[1:])
ind = list(rawIndicators.index.values)

merged = rawIndicators.merge(rawContinents, left_index=True, right_index=True) \
.drop('Country_y', axis=1).rename(columns={'Country_x': 'Country'})[['Continent', 'Country'] + labInd]

# A1
labValAdaugata = list(merged.columns.values[-7:])
merged[['Country'] + labValAdaugata] \
.set_index('Country', append=True) \
.sum(axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged[['Continent'] + labInd] \
.groupby('Continent') \
.apply(func=lambda df: pd.Series({ind: np.round(np.std(df[ind]) / np.mean(df[ind]) * 100, 2) for ind in labInd})) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = merged[labInd].values
def standardize(x: np.ndarray):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x - means) / stds
x = standardize(x)

cov = np.cov(x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
print(alpha)

# B2
a = eigenvectors[:, k_desc]
C = x @ a
Rxc = a * np.sqrt(alpha)
scores = C / np.sqrt(alpha)
pd.DataFrame(data=np.round(scores, 2), index=ind, columns=labInd).to_csv('./dataOUT/scoruri.csv')

# B3
plt.figure(figsize=(12, 9))
plt.title('Corelograma scorurilor')
sb.heatmap(scores, vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()

# C
factorLoadings = pd.read_csv('./dataIN/g20.csv', index_col=0)
communalities = np.cumsum(factorLoadings * factorLoadings, axis=1)
print(communalities.sum().idxmax())