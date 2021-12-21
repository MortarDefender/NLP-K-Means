import json
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from main import getLabels, getFeatureVectors

class KmeansPlus:
    def __init__(self, numOfClusters, fileName):
        self.__K = numOfClusters
        self.__vectors = getFeatureVectors(fileName)
        self.__centroids = [0 for i in range(self.__K)]
        self.__distances = [1 for i in enumerate(self.__vectors)]
        self.__weightedDistances = [1 for i in enumerate(self.__vectors)]

    def __placeClusterCentroids(self):
        return (random.choices(self.__vectors, weights=(self.__weightedDistances), k=1))[0]
    
    def __computeDistance(self, x, y):
        return distance.euclidean(x, y)
    
    def plot(self):
        pass
    
    def fit(self):
        self.__centroids[0] = self.__placeClusterCentroids(self)
        for i, point in enumerate(self.__vectors):
            self.__distances[i] = self.__computeDistance(point[0], self.__centroids[0])
            self.__weightedDistances[i] = pow(self.__distances[i],2)

        for i in range(1,self.__K):
            self.__centroids[i] = self.__placeClusterCentroids(self)
            for j, point in enumerate(self.__vectors):
                D = self.__computeDistance(point[0],self.__centroids[i])
                if D < self.__distances[j]:
                    self.__distances[j] = D
                    self.__weightedDistances[j] = pow(D,2)