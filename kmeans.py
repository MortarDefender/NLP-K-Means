import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
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
    
    @staticmethod
    def getLabels(fileName):
        allLabels = []
        
        with open(fileName, "r") as f:
            tsv = csv.reader(f, delimiter="\t")
            
            for line in tsv:
                allLabels.append(line[0])
        
        return allLabels, len(set(allLabels))
    
    def __placeClusterCentroids(self):
        randomCentroids = np.random.permutation(self.__data.shape[0])[:self.__numberOfClusters]
        self.__centers = self.__data[randomCentroids]

    def __findNearstCentrois(self, dataPoint):
        # minCenterIndex = 0
        # minDistance = self.__checkDistance(dataPoint, self.__centers[0]) 
        
        # for i, center in enumerate(self.__centers):
        #     currentDistance = self.__checkDistance(dataPoint, center) 
        #     if minDistance < currentDistance:
        #         minDistance = currentDistance
        #         minCenterIndex = i
        
        # return minCenterIndex
        
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
        plt.scatter(self.__data[:, 0], self.__data[:, 1], marker = '.',
                color = 'gray', label = 'data points')
        plt.scatter(self.__centers[:-1, 0], self.__centers[:-1, 1],
                    color = 'black', label = 'selected centroids')
        plt.title('Select % d th centroid' % (self.__centers.shape[0]))
        
        plt.legend()
        plt.xlim(-5, 12)
        plt.ylim(-10, 15)
        plt.show()
    
    def accuracy(self, trueLabels):
        return rand_score(trueLabels, self.__clusterLabels), adjusted_rand_score(trueLabels, self.__clusterLabels), 
    
    def predict(self, data):
        return self.__findNearstCentrois(data)
    
    def fit(self, data):
        index = 0
        self.__data = data
        self.__placeClusterCentroids()
        
        while index < self.__maxNumberOfIterations and not self.__checkConvergence():
            self.__clusterLabels = self.__findNearstCentrois(self.__data)
            
            for i in range(self.__numberOfClusters):
                self.__center[i] = self.__data[self.__clusterLabels == i].mean(axis = 0)
            
            
            if len(self.__pastDataSets) > 2:
                del self.__pastDataSets[0]

            self.__pastDataSets.append(self.__clusterLabels)
            
            index += 1
