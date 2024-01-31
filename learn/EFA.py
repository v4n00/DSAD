import factor_analyzer as fa
import pandas as pd

x_df = pd.DataFrame() # standardized

kmo = fa.calculate_kmo(x_df)
# kmo[1] trb sa fie > 0.6

EFAModel = fa.FactorAnalyzer(n_factors=len(x_df.columns.values))
EFAModel.fit(x_df)
factorLoadings = EFAModel.loadings_ # aka common factors
specificFactors = EFAModel.get_uniquenesses()
eigenValues = EFAModel.get_eigenvalues() # aka principal components