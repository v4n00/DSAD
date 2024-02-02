import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo

x_df = pd.DataFrame() # standardized

kmo = calculate_kmo(x_df)
# kmo[1] trb sa fie > 0.6

EFAModel = FactorAnalyzer(n_factors=len(x_df.columns.values))
scores = EFAModel.fit_transform(x_df)
factorLoadings = EFAModel.loadings_ # aka common factors
specificFactors = EFAModel.get_uniquenesses()
eigenValues = EFAModel.get_eigenvalues() # aka principal components