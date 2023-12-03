import numpy as np
from numpy.random import rand
from tslearn.metrics import dtw

EUCLIDEAN = 0
DTW = 1
DIST_FUNC = EUCLIDEAN

__DEBUG__ = True

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
    centroids = np.zeros((k,d))
    emptyClusters = []
    for index, data in enumerate(clusters):
        n = len(data)
        if n == 0:
            emptyClusters.append(index)
            continue
        totalSum = np.sum(data, axis=0)
        mean = totalSum / n
        centroids[index,:] = mean
    return centroids, emptyClusters

def getDistance(trainingSet, mean):
    '''
    Distance formula between training set and a cluster's mean
    trainingSet: (n, d=24), n = # participants, d = # features/hours
    mean: (d=24, 1), column  vector
    distance: (n, 1)
    '''
   
    if DIST_FUNC == EUCLIDEAN:
        # Euclidean distance formula: 
        diff_sq = np.square(trainingSet - mean)
        distance = np.sqrt(np.sum(diff_sq, axis=1))
        distance = np.reshape(distance, (distance.shape[0],))
        return distance
    else:
        # Dynamic time warping distance formula:
        dtw_distance = []
        for i in range(len(trainingSet)):
            dtw_distance.append(dtw(trainingSet[i], mean))
        return dtw_distance

def getObjective(clusters, centroids):
    objective = 0
    for index, data in enumerate(clusters):
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

    #data = np.array(data)
    n, d = data.shape # number of participants, number of features

    clusters = [0] * k # cluster index : datasets for each cluster
    clusters = np.array(clusters, dtype=object)

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

        centroids, emptyClusters = newCentroid(clusters, k, d)

        for j in range(len(emptyClusters)-1, -1, -1): # changes objective
            k -= 1
            index = emptyClusters[j]
            clusters = np.delete(clusters, index, axis=0)
            centroids = np.delete(centroids, index, axis=0)
    
        objectives.append(getObjective(clusters, centroids))

    if __DEBUG__:
        print(k, " clusters")
        print("Centroids:")
        for mean in centroids:
            print(mean)
        for i, cluster_data in enumerate(clusters):
            print(i, " : ", len(cluster_data))

    return centroids, clusters, objectives
