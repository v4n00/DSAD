import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo

x = np.ndarray() # standardized

kmo = calculate_kmo(x) # kmo[1] trb sa fie > 0.6

efa = FactorAnalyzer(n_factors=x.shape[1] - 1) # nr de factori e nr de coloane - 1
scores = efa.fit_transform(x)
factorLoadings = efa.loadings_
eigenvalues = efa.get_eigenvalues()
communalities = efa.get_communalities()
specificFactors = efa.get_uniquenesses()