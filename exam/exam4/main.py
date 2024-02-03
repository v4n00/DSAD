import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2024 Furtuna
# dataset is generated with ChatGPT
rawRata = pd.read_csv('./dataIN/Rata.csv', index_col=0)
rawCoduri = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labels = list(rawRata.columns[1:].values)

merged = rawRata.merge(rawCoduri, left_index=True, right_index=True) \
.drop('Country_Name', axis=1)[['Continent', 'Country'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged[merged['RS'] < np.average(merged['RS'])][['Country', 'RS']] \
.sort_values('RS', ascending=False) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
merged \
.set_index('Country') \
.groupby('Continent') \
.apply(func=lambda df: pd.Series({rata: df[rata].idxmax() for rata in labels})) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
pve = pca.explained_variance_ratio_

var_cum = np.cumsum(alpha)
pve_cum = np.cumsum(pve)

pd.DataFrame(data={'Varianta componentelor': alpha,
                    'Varianta cumulata': var_cum,
                    'Procentul de varianta explicata': pve, 
                    'Procentul cumulat': pve_cum}) \
.to_csv('./dataOUT/Varianta.csv')

# B2
plt.figure(figsize=(10, 10))
plt.title('Varianta explicata de catre componente')
labels = ['C' + str(i + 1) for i in range(len(alpha))]
plt.plot(labels, alpha, 'bo-')
plt.axhline(1, c='r')
plt.show()

# B3
a = pca.components_.T
Rxc = a * np.sqrt(alpha)
communalities = np.cumsum(Rxc * Rxc, axis=1)
communalities_df = pd.DataFrame(data=communalities, index=labels, columns=['C' + str(i + 1) for i in range(communalities.shape[1])])

plt.figure(figsize=(10, 10))
plt.title('Corelograma corelatilor')
sb.heatmap(communalities_df, vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()

# C
# I don't know