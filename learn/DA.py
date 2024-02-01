import pandas as pd
import sklearn.discriminant_analysis as skl
import sklearn.model_selection as sklt

x = pd.DataFrame() # standardized
x_applied = pd.DataFrame()

tinta = 'VULNERAB' # coloana specificata in cerinta
x_train, x_test, y_train, y_test = sklt.train_test_split(x, x[tinta], train_size=0.4)
model = skl.LinearDiscriminantAnalysis()
model.fit(x_train, y_train)

scores = model.transform(x_test)
prediction_test = model.predict(x_test)
prediction_applied = model.predict(x_applied.loc[:, x_applied.columns != tinta])