import numpy as np
from numpy.random import rand

def initKRandomCentroids(trainingSet, k):
    '''
    trainingSet: 'row' participants x 'col' hours (24)
    initial_centroids: (k centroids, d=24)
        each randomly assigned to a participant's data
    '''
    row, col = trainingSet.shape
    indices = np.random.randint(row, size=k)
    initial_centroids = trainingSet[indices, :]
    return initial_centroids

def newCentroid(clusters, k, d):
    # What to do if no data in a cluster?
    centroids = np.zeros((k,d))
    for index, data in clusters.items():
        n = len(data)
        mean = np.sum(data, axis=0) #/ n
        print(n)
        mean = mean / n
        centroids[index,:] = mean
    return centroids

    
# To do: change to dynamic time warping (DTW)
def getDistance(trainingSet, mean):
    '''
    Euclidean distance formula between training set and a cluster's mean
    trainingSet: (n, d=24), n = # participants, d = # features/hours
    mean: (d=24, 1), column  vector
    distance: (n, 1)
    '''
    diff_sq = np.square(trainingSet - mean)
    distance = np.sqrt(np.sum(diff_sq, axis=1))
    distance = np.reshape(distance, (distance.shape[0],))
    return distance

def getObjective(clusters, centroids):
    objective = 0
    for index, data in clusters.items():
        diff_sq = np.square(data - centroids[index, :])
        objective += np.sum(diff_sq)
    return objective

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
    n, d = data.shape # number of participants, number of features

    clusters = {} # cluster index : datasets for each cluster
    centroids = initKRandomCentroids(data, k)
    objectives = []
    for i in range(max_iter):
        distances = np.zeros((n, k))
        for j, mean in enumerate(centroids):
            distance = getDistance(data, mean)
            distances[:, j] = distance
        minDist = np.argmin(distances, axis=1)

        # Find training points closest to cluster and assign them to cluster
        for j in range(k):
            indices = np.where(minDist == j)
            clusters[j] = data[indices]
        centroids = newCentroid(clusters, k, d)
        objectives.append(getObjective(clusters, centroids))
    return centroids, clusters, objectives
