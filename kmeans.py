import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Kmeans:
    def __init__(self, numberOfClusters = 0):
        self.__numberOfClusters = numberOfClusters
        self.__maxNumberOfIterations = 100
        self.__pastDataSets = []
        self.__centers = []

    def __placeClusterCentroids(self, data):
        self.__centers = np.random.choice(data, size = self.__numberOfClusters)

    def __findNearstCentrois(self, dataPoint):
        minCenterIndex = 0
        minDistance = self.__checkDistance(dataPoint, self.__centers[0]) 
        
        for i, center in enumerate(self.__centers):
            currentDistance = self.__checkDistance(dataPoint, center) 
            if minDistance < currentDistance:
                minDistance = currentDistance
                minCenterIndex = i
        
        return minCenterIndex
    
    def __checkDistance(self, pointA, pointB):
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
        pass
    
    def fit(self, data):
        index = 0
        self.__placeClusterCentroids(data)
        
        while index < self.__maxNumberOfIterations or self.__checkConvergence():
            
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
