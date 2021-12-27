import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score


class Kmeans:
    def __init__(self, numberOfClusters = 0):
        self.__numberOfClusters = numberOfClusters
        self.__maxNumberOfIterations = 100
        self.__clusterLabels = None
        self.__pastDataSets = []
        self.__centers = []
        self.__data = None
    
    def __placeClusterCentroids(self):
        """ create random clusters """
        
        randomCentroids = np.random.permutation(self.__data.shape[0])[: self.__numberOfClusters]
        self.__centers = self.__data[randomCentroids]

    def __findNearstCentrois(self, dataPoint):
        """ find the distance between all the data points to all the centeres """
        
        if dataPoint.ndim == 1:
            dataPoint = dataPoint.reshape(-1, 1)
        
        centroidDistances =  pairwise_distances(dataPoint, self.__centers, metric = 'euclidean')
        self.__clusterLabels = np.argmin(centroidDistances, axis = 1)
        return self.__clusterLabels
    
    @staticmethod
    def __checkDistance(pointA, pointB):
        """ calculate the distance between two points """
        
        return distance.euclidean(pointA, pointB)

    def plot(self):
        """ plot the centers and the given data points """
        
        sklearn_pca = PCA(n_components = 2)
        t_labels = sklearn_pca.fit_transform(self.__data)
        plt.scatter(t_labels[:, 0], t_labels[:, 1], c = self.__clusterLabels, s = 50, cmap = 'viridis')
        plt.scatter(self.__centers[:, 0], self.__centers[:, 1],c = 'black', s = 300, alpha = 0.6)
        plt.show()
    
    def accuracy(self, trueLabels):
        """ return the accuracy of the model in the form of (ri score, ari score) """
        
        return rand_score(trueLabels, self.__clusterLabels), adjusted_rand_score(trueLabels, self.__clusterLabels)
    
    def fit(self, data):
        """ main function: create the clusteres and over x iterations improve the labels of the data points """
        
        self.__data = data
        self.__placeClusterCentroids()
        
        for index in range(self.__maxNumberOfIterations):
            self.__clusterLabels = self.__findNearstCentrois(self.__data)
            
            for i in range(self.__numberOfClusters):
                clusterIthData = self.__data[self.__clusterLabels == i]
                
                if len(clusterIthData) == 0: # ignore empty slice
                    continue
                
                self.__centers[i] = clusterIthData.mean(axis = 0)
            
            
            if len(self.__pastDataSets) > 2:
                del self.__pastDataSets[0]

            self.__pastDataSets.append(self.__clusterLabels)
