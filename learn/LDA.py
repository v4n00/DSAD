import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

x = pd.DataFrame() # DOES NOT need to be standardized
x_applied = pd.DataFrame() # DOES NOT need to be standardized

tinta = 'VULNERAB' # column specified in the requirements
variabile = list(x.columns.values[:-1]) # the other columns

x_train, x_test, y_train, y_test = train_test_split(x[variabile], x[tinta], train_size=0.4)
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train) # trains the model

scores = lda.transform(x_test)
prediction_test = lda.predict(x_test)
prediction_applied = lda.predict(x_applied)