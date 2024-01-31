import numpy as np


def standardise(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x - means) / stds

def replaceNAN(x):
    means = np.nanmean(x, axis=0)
    locs = np.where(np.isnan(x))
    x[locs] = means[locs[1]]
    return x