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
        self.__pastDataSets = []
        self.__centers = []
        self.__data = None
        self.__clusterLabels = None
    
    def __placeClusterCentroids(self):
        randomCentroids = np.random.permutation(self.__data.shape[0])[:self.__numberOfClusters]
        self.__centers = self.__data[randomCentroids]

    def __findNearstCentrois(self, dataPoint):
        if dataPoint.ndim == 1:
            dataPoint = dataPoint.reshape(-1, 1)
        
        centroidDistances =  pairwise_distances(dataPoint, self.__centers, metric = 'euclidean')
        self.__clusterLabels = np.argmin(centroidDistances, axis = 1)
        return self.__clusterLabels
    
    @staticmethod
    def __checkDistance(pointA, pointB):
        # euclidean distance
        return distance.euclidean(pointA, pointB)

    def __checkConvergence(self):
        points = 0
        
        for i, firstSet in enumerate(self.__pastDataSets):
            for j, secondSet in enumerate(self.__pastDataSets):
                if i < j:
                    if firstSet == secondSet:
                        points += 1
                        
        return points >= 2

    def plot(self):
        sklearn_pca = PCA(n_components = 2)
        t_labels = sklearn_pca.fit_transform(self.__data)
        plt.scatter(t_labels[:, 0], t_labels[:, 1], c=self.__clusterLabels, s=50, cmap='viridis')
        plt.scatter(self.__centers[:, 0], self.__centers[:, 1],c='black', s=300, alpha=0.6)
        plt.show()
        
        # print(self.__centers)
    
    def accuracy(self, trueLabels):
        # print(trueLabels)
        # print("!!!!!!!!")
        # print(self.__clusterLabels)
        return rand_score(trueLabels, self.__clusterLabels), adjusted_rand_score(trueLabels, self.__clusterLabels)
    
    def predict(self, data):
        return self.__findNearstCentrois(data)
    
    def fit(self, data):
        # index = 0
        self.__data = data
        self.__placeClusterCentroids()
        
        # while index < self.__maxNumberOfIterations and not self.__checkConvergence():
        for index in range(self.__maxNumberOfIterations):
            self.__clusterLabels = self.__findNearstCentrois(self.__data)
            # print(self.__clusterLabels)
            
            for i in range(self.__numberOfClusters):
                self.__centers[i] = self.__data[self.__clusterLabels == i].mean(axis = 0)
            
            
            if len(self.__pastDataSets) > 2:
                del self.__pastDataSets[0]

            self.__pastDataSets.append(self.__clusterLabels)
            
            # index += 1
