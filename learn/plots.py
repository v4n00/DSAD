import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from seaborn import heatmap, kdeplot


def correlogram(x):
    plt.figure(figsize=(15, 11))
    plt.title('Correlogram')
    heatmap(data=x, vmin=-1, vmax=1, cmap='bwr', annot=True)

def linePlot(alpha):
    plt.figure(figsize=(11, 8))
    plt.title('Line plot')
    Xindex = ['C' + str(k + 1) for k in range(len(alpha))]
    plt.plot(Xindex, alpha, 'bo-')
    plt.axhline(1, color='r')

def biplot(x, y):
    plt.figure(figsize=(7, 7))
    plt.title('Biplot CCA')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x[:, 0], x[:, 1], c='r', label='X')
    plt.scatter(y[:, 0], y[:, 1], c='b', label='Y')
    plt.legend()

def dendrogram(h, labels, threshold):
    plt.figure(figsize=(15, 8))
    plt.title('Clusters')
    dendrogram(h, labels=labels, leaf_rotation=30)
    plt.axhline(threshold, c='r')

def correlationCircle(data):
    plt.figure(figsize=(12, 12))
    plt.title('Correlation circle')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(0, c='g')
    plt.axvline(0, c='g')
    plt.scatter(data[:, 0], data[:, 1])

def kdeplot(scores):
    plt.figure(figseize=(12, 12))
    plt.title('Scores')
    kdeplot(scores, fill=True)