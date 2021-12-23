import json
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class KmeansPlus:
    def __init__(self, numOfClusters, vectors):
        self.__K = numOfClusters
        self.__vectors = vectors
        self.__centroids = [0 for i in range(self.__K)]
        self.__distances = [[1,0] for i in enumerate(self.__vectors)]
        self.__weightedDistances = [1 for i in enumerate(self.__vectors)]
        self.__vecInCentroid = [[] for i in range(self.__K)]

    def __placeClusterCentroids(self):
        return (random.choices(self.__vectors, weights=(self.__weightedDistances), k=1))[0]
    
    def __computeDistance(self, x, y):
        return distance.euclidean(x, y)
    
    def plot(self):
        plt.scatter(self.__vectors[:, 0], self.__vectors[:, 1], marker='.',
                    color='gray', label='data points')
        plt.scatter(self.__centroids[:-1, 0], self.__centroids[:-1, 1],
                    color='black', label='selected centroids')
        plt.title('Select % d th centroid' % (self.__centroids.shape[0]))

        plt.legend()
        plt.xlim(-5, 12)
        plt.ylim(-10, 15)
        plt.show()
    
    def fit(self):
        self.__centroids[0] = self.__placeClusterCentroids()
        for i, point in enumerate(self.__vectors):
            self.__distances[i][0] = self.__computeDistance(point[0], self.__centroids[0])
            self.__weightedDistances[i] = pow(self.__distances[i][0],2)

        for i in range(1,self.__K):
            self.__centroids[i] = self.__placeClusterCentroids()
            for j, point in enumerate(self.__vectors):
                D = self.__computeDistance(point[0],self.__centroids[i])
                if D < self.__distances[j][0]:
                    self.__distances[j][1] = i
                    self.__distances[j][0] = D
                    self.__weightedDistances[j] = pow(D,2)

        for vec in self.__distances:
            self.__vecInCentroid[vec[1]].append(vec[0])