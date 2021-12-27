import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score


class KmeansPlus:
    def __init__(self, numOfClusters):
        self.__vectors = None
        self.__centroids = None
        self.__distances = None
        self.__K = numOfClusters
        self.__weightedDistances = None

    def __reFit(self, data):
        """ set the class variables to the new data """
        
        self.__centroids = []
        self.__vectors = data
        self.__distances = [[np.inf, 0] for i in range(len(self.__vectors))]
        self.__weightedDistances = [1 for i in enumerate(self.__vectors)]

    def __placeClusterCentroids(self):
        """ create a randome center using weights from all the data points """
        
        return random.choices(self.__vectors, weights=(self.__weightedDistances), k = 1)

    def __computeDistance(self, dataPoint, cent):
        """ check the distance between all data points to the last center added
            and insert the distance squared into the weights """
        
        distances = pairwise_distances(dataPoint, self.__centroids[-1], metric = 'euclidean')
        
        for i, dis in enumerate(distances):
            if self.__distances[i][0] > dis:
                self.__distances[i] = [dis, cent]
                self.__weightedDistances[i] = dis ** 2

    def accuracy(self, trueLabels):
        """ return the accuracy of the model in the form of (ri score, ari score) """
        
        labels = [item[1] for item in self.__distances]
        return rand_score(trueLabels, labels), adjusted_rand_score(trueLabels, labels)

    def plot(self):
        """ plot the centeres and all the data points given """
        
        cent = np.array(self.__centroids)
        labels = [item[1] for item in self.__distances]
        sklearn_pca = PCA(n_components = 2)
        t_labels = sklearn_pca.fit_transform(self.__vectors)
        plt.scatter(t_labels[:, 0], t_labels[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(cent[0][:, 0], cent[0][:, 1],c='black', s=300, alpha=0.6)
        plt.show()
    
    def fit(self, data):
        """ create all centeres over k iterations and improve the centeres locations """
        
        self.__reFit(data)
       
        for i in range(self.__K):
            self.__centroids.append(self.__placeClusterCentroids())
            self.__computeDistance(self.__vectors, i)
