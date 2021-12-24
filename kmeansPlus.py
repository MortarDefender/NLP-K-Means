import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, rand_score, adjusted_rand_score


class KmeansPlus:
    def __init__(self, numOfClusters, vec):
        self.__K = numOfClusters
        self.__vectors = vec
        self.__centroids = []
        self.__distances = [[np.inf,0] for i in range(len(vec))]
        self.__weightedDistances = [1 for i in enumerate(self.__vectors)]

    def __reFit(self):
        self.__centroids = []
        self.__distances = [[np.inf, 0] for i in range(len(self.__vectors))]
        self.__weightedDistances = [1 for i in enumerate(self.__vectors)]

    def __placeClusterCentroids(self):
        return random.choices(self.__vectors, weights=(self.__weightedDistances), k=1)

    def __computeDistance(self, dataPoint):
        distances = pairwise_distances(dataPoint, self.__centroids[-1], metric='euclidean')
        for i, dis in enumerate(distances):
            if self.__distances[i][0] > dis:
                self.__distances[i] = [dis, len(self.__centroids) - 1]
                self.__weightedDistances[i] = dis ** 2

    def accuracy(self, trueLabels):
        labels = [item[1] for item in self.__distances]
        return rand_score(trueLabels, labels), adjusted_rand_score(trueLabels, labels)

    def plot(self):
        cent = np.array(self.__centroids)
        labels = [item[1] for item in self.__distances]
        sklearn_pca = PCA(n_components = 2)
        t_labels = sklearn_pca.fit_transform(self.__vectors)
        plt.scatter(t_labels[:, 0], t_labels[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(cent[0][:, 0], cent[0][:, 1],c='black', s=300, alpha=0.6)
        plt.show()
    
    def fit(self):
        self.__reFit()
        for i in range(self.__K):
            self.__centroids.append(self.__placeClusterCentroids())
            self.__computeDistance(self.__vectors)
