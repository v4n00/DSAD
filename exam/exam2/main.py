import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subject exam 2024 Furtuna
# dataset is generated with ChatGPT
rawIndicators = pd.read_csv('./dataIN/GlobalIndicatorsPerCapita_2021.csv', index_col=0)
rawContinents = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labels = list(rawIndicators.columns.values[1:])
indexes = list(rawIndicators.index.values)

merged = rawIndicators.merge(rawContinents, left_index=True, right_index=True) \
.drop('Country_y', axis=1).rename(columns={'Country_x': 'Country'})[['Continent', 'Country'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
labValAdaugata = list(merged.columns.values[-7:])
merged[['Country'] + labValAdaugata] \
.set_index('Country', append=True) \
.sum(axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
# this is correct, it uses the population formula for the standard deviation,
# if you do DataFrame.std() it uses the sample formula for the standard deviation,
# if you want to do that, use argument DataFrame.std(ddof=0)
merged[['Continent'] + labels] \
.groupby('Continent') \
.apply(func=lambda df: pd.Series({ind: np.round(np.std(df[ind]) / np.mean(df[ind]) * 100, 2) for ind in labels})) \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print(alpha)

# B2
scores = C / np.sqrt(alpha)
pd.DataFrame(data=np.round(scores, 2), index=indexes, columns=labels).to_csv('./dataOUT/scoruri.csv')

# B3
plt.figure(figsize=(12, 9))
plt.title('Scoruri')
plt.scatter(scores[:, 0], scores[:, 1])
plt.show()

# C
factorLoadings = pd.read_csv('./dataIN/g20.csv', index_col=0)
communalities = np.cumsum(factorLoadings * factorLoadings, axis=1)
print(communalities.sum().idxmax())