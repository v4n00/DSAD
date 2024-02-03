import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# ᗜˬᗜ - subject exam 2023/2024 Furtuna
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
tinta = 'VULNERAB' # the column that we need to predict
labels = list(x.columns.values[:-1]) # rest of the columns

# map the letter to numbers
dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
x[tinta] = x[tinta].map(dict)

# function takes as a first argument the data set without the target,
# the 2nd argument is only the column that we need to predict
# splits the datasets into 2 sets that each split into another 2,
# one for training and one for testing,
# into one with everything without the target, and one with only the target
x_train, x_test, y_train, y_test = train_test_split(x[labels], x[tinta], train_size=0.4)
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train) # trains the model
scores = model.transform(x_train) # only gives the scores

pd.DataFrame(data=scores) \
.to_csv('./dataOUT/z.csv')

# B2
plt.figure(figsize=(12, 12))
plt.title('Scoruri')
sb.kdeplot(scores, fill=True)
plt.show()

# B3
x_apply = pd.read_csv('./dataIN/ProiectB_apply.csv', index_col=0)

prediction_test = model.predict(x_test) # predict on the dataset for testing
prediction_applied = model.predict(x_apply) # predict on the extra dataset given to us

pd.DataFrame(data=prediction_test) \
.to_csv('./dataOUT/predict_test.csv')
pd.DataFrame(data=prediction_applied) \
.to_csv('./dataOUT/predict_apply.csv')