import numpy as np
from numpy.random import rand

def initKRandomCentroids(trainingSet, k):
    '''
    trainingSet: 'row' participants x 'col' hours (24)
    initial_centroids: k centroids, each randomly assigned to a participant's data
    '''
    row, col = trainingSet.shape
    indices = np.random.randint(row, size=k)
    initial_centroids = trainingSet[indices, :]
    return initial_centroids

# To do: change to dynamic time warping (DTW)
def getDistance(trainingSet, mean):
    '''
    Euclidean distance formula
    '''
    diff_sq = np.square(trainingSet - mean)
    distance = np.sqrt(np.sum(diff_sq, axis=1))

def kmeans(data, k, max_iter):
    '''
    K Means Algorithm:
    * Intialize K cluster centroids: random initialization
    * For 'max_iter' iterations:
        * For each data point in dataset/training set:
            * Calculate distance to each K centroids: euclidean distance
            * Reassign data point to cluster with nearest centroid
        * Update centroids for each cluster
        * Check for convergence (objective function)
            * Stop if cluster did not change much from previous cluster
    * Return clusters
    '''
    data = np.array(data)

    clusters = {} # cluster index : datasets for each cluster
    centroids = initKRandomCentroids(data, k)

    for i in range(max_iter):
        for mean in centroids:
            getDistance(data, mean)
            
