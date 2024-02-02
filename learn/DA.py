import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

x = pd.DataFrame() # NU TREBUIE STANDARDIZAT
x_applied = pd.DataFrame() # NU TREBUIE STANDARDIZAT

variabile = list(x.columns.values[:-1]) # toate coloanele inafara de tinta
tinta = 'VULNERAB' # coloana specificata in cerinta

x_train, x_test, y_train, y_test = train_test_split(x[variabile], x[tinta], train_size=0.4)
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train) # train la model

scores = model.transform(x_test)
prediction_test = model.predict(x_test)
prediction_applied = model.predict(x_applied)