import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import rand_score
from sklearn.metrics.cluster import adjusted_rand_score


class Kmeans:
    def __init__(self, numberOfClusters = 0):
        self.__numberOfClusters = numberOfClusters
        self.__maxNumberOfIterations = 100
        self.__pastDataSets = []
        self.__centers = []
        self.__data = None
    
    @staticmethod
    def getLabels(fileName):
        allLabels = []
        
        with open(fileName, "r") as f:
            tsv = csv.reader(f, delimiter="\t")
            
            for line in tsv:
                allLabels.append(line[0])
        
        return allLabels, len(set(allLabels))
    
    def __placeClusterCentroids(self):
        self.__centers = np.random.choice(self.__data, size = self.__numberOfClusters)

    def __findNearstCentrois(self, dataPoint):
        minCenterIndex = 0
        minDistance = self.__checkDistance(dataPoint, self.__centers[0]) 
        
        for i, center in enumerate(self.__centers):
            currentDistance = self.__checkDistance(dataPoint, center) 
            if minDistance < currentDistance:
                minDistance = currentDistance
                minCenterIndex = i
        
        return minCenterIndex
    
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
        plt.title('Select % d th centroid'%(self.__centers.shape[0]))
         
        plt.legend()
        plt.xlim(-5, 12)
        plt.ylim(-10, 15)
        plt.show()
    
    def accuracy(self):
        pass
    
    def fit(self, data):
        index = 0
        self.__data = data
        self.__placeClusterCentroids()
        
        while index < self.__maxNumberOfIterations and not self.__checkConvergence():
            
            dataPoints = [[]] * self.__numberOfClusters
            
            for point in data:
                center = self.__findNearstCentrois(point)
                dataPoints[center].append(point)
            
            for i, cluster in enumerate(self.__dataPoints):
                self.__center[i] = np.mean(cluster)
            
            if len(self.__pastDataSets) > 2:
                del self.__pastDataSets[0]

            self.__pastDataSets.append(dataPoints)
            
            index += 1
