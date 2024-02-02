import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

x = pd.DataFrame() # NU TREBUIE STANDARDIZAT
x_applied = pd.DataFrame() # NU TREBUIE STANDARDIZAT

tinta = 'VULNERAB' # coloana specificata in cerinta
x_train, x_test, y_train, y_test = train_test_split(x.loc[:, x.columns != tinta], x[tinta], train_size=0.4)
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)

scores = model.transform(x_test)
prediction_test = model.predict(x_test)
prediction_applied = model.predict(x_applied)