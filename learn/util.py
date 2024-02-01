import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

x = pd.DataFrame()
labels = list() # list of labels for variables

# replace NAN 
x.fillna(np.mean(x[labels], axis=0), inplace=True)

# standardize
x = StandardScaler().fit_transform(x)