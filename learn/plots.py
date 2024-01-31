import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hic
import seaborn as sb


def correlogram(C, title='Correlogram'):
    plt.figure(figsize=(15, 11))
    plt.title(title)
    sb.heatmap(data=np.round(C, 2), vmin=-1, vmax=1, cmap='bwr', annot=True)

def linePlot(alpha, title='Line plot'):
    plt.figure(figsize=(11, 8))
    plt.title(title)
    Xindex = ['C' + str(k + 1) for k in range(len(alpha))]
    plt.plot(Xindex, alpha)
    plt.axhline(1, color='r')

def biplot(x, y, title='Biplot'):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x[:, 0], x[:, 1], c='r', label='X')
    plt.scatter(y[:, 0], y[:, 1], c='b', label='Y')
    plt.legend()

def dendrogram(h, labels, threshold, title='Hierarchical Clusters'):
    plt.figure(figsize=(15, 8))
    plt.title(title)
    hic.dendrogram(h, labels=labels, leaf_rotation=30)
    plt.axhline(threshold, c='r')