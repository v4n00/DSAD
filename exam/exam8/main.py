import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.discriminant_analysis as skl
import sklearn.model_selection as sklt

ind = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
pop = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)
labInd = list(ind.columns[1:].values)

merged = ind.merge(right=pop, right_index=True, left_index=True) \
.drop(columns='Localitate_y') \
.rename(columns={'Localitate_x' : 'Localitate'})[['Judet', 'Localitate', 'Populatie'] + labInd]

# A1
merged[['Localitate', 'Populatie'] + labInd] \
.apply(lambda row: row[labInd] / row['Populatie'], axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
r2 = merged[['Judet'] + labInd].groupby('Judet').sum()
r2['Cifra Afaceri'] = r2.max(axis=1)
r2['Activitate'] = r2.idxmax(axis=1)
r2[['Cifra Afaceri','Activitate']] \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = pd.read_csv('./dataIN/ProiectB.csv')
tinta = 'VULNERAB'
dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

x[tinta] = x[tinta].map(dict)

x_train, x_test, y_train, y_test = sklt.train_test_split(x, x[tinta], train_size=0.4)
model = skl.LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
scores = model.transform(x_test)

pd.DataFrame(data=scores) \
.to_csv('./dataOUT/z.csv')

# B2
plt.figure(figsize=(12, 12))
plt.title('Corelograma scorurilor')
sb.heatmap(np.round(scores, 2), vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()

# B3
prediction_test = model.predict(x_test)
prediction_applied = model.predict(x)

pd.DataFrame(data=prediction_test) \
.to_csv('./dataOUT/predict_test.csv')
pd.DataFrame(data=prediction_applied) \
.to_csv('./dataOUT/predict_apply.csv')