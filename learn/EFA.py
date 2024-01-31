import factor_analyzer as fa
import pandas as pd

x_df = pd.DataFrame() # standardized

kmo = fa.calculate_kmo(x_df)
print(kmo)
# incepand cu al doilea numar din array-ul kmo,
# numaram cate numere > 0.60 pana la primul numar < 0.60
# exemplu: [0.20, 0.64, 0.69, 0.68, 0.66, 0.64, 0.59, 0.24, ...] => 5 factori
noFactors = 5

EFAModel = fa.FactorAnalyzer(n_factors=noFactors)
EFAModel.fit(x_df)
factorLoadings = EFAModel.loadings_ # aka common factors
specificFactors = EFAModel.get_uniquenesses()
eigenValues = EFAModel.get_eigenvalues() # aka principal components