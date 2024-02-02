import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ᗜˬᗜ - subiect examen furtuna 2023
ind = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
pop = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)
labels = list(ind.columns[1:].values)

merged = ind.merge(right=pop, right_index=True, left_index=True) \
.drop(columns='Localitate_y') \
.rename(columns={'Localitate_x' : 'Localitate'})[['Judet', 'Localitate', 'Populatie'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1
merged[['Localitate', 'Populatie'] + labels] \
.apply(lambda row: row[labels] / row['Populatie'], axis=1) \
.to_csv('./dataOUT/Cerinta1.csv')

# A2
r2 = merged[['Judet'] + labels].groupby('Judet').sum()
r2['Cifra Afaceri'] = r2.max(axis=1)
r2['Activitate'] = r2.idxmax(axis=1)
r2[['Cifra Afaceri','Activitate']] \
.to_csv('./dataOUT/Cerinta2.csv')

# B1
x = pd.read_csv('./dataIN/ProiectB.csv', index_col=0)
labels = list(x.columns.values[:-1]) # restul de coloane
tinta = 'VULNERAB' # coloana care trb sa prezicem

# mapam literele in numere
dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
x[tinta] = x[tinta].map(dict)

# functia primeste ca prim argument data set-ul fara tina, si al 2-lea argument doar coloana de tinta
# imparte datasetu in 2 set-uri, fiecare are 1 set de train si unul de test
x_train, x_test, y_train, y_test = train_test_split(x[labels], x[tinta], train_size=0.4)
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train) # invata modelul
scores = model.transform(x_train) # doar da scoruri

pd.DataFrame(data=scores) \
.to_csv('./dataOUT/z.csv')

# B2
plt.figure(figsize=(12, 12))
plt.title('Scoruri')
sb.kdeplot(scores, fill=True)
plt.show()

# B3
x_apply = pd.read_csv('./dataIN/ProiectB_apply.csv', index_col=0)

prediction_test = model.predict(x_test) # predict pe dataset-ul de testat
prediction_applied = model.predict(x_apply) # predict pe dataset-ul de aplicare (este acelasi)

pd.DataFrame(data=prediction_test) \
.to_csv('./dataOUT/predict_test.csv')
pd.DataFrame(data=prediction_applied) \
.to_csv('./dataOUT/predict_apply.csv')