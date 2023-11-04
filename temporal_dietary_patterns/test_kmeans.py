from parse_dataset import *
from kmeans_clustering import *

import random
import matplotlib.pyplot as plt

# Make fake dataset
random.seed(10)
rng = np.random.default_rng(0)
n_per_cluster = 50
xy1 = rng.multivariate_normal([2,2], [[0.02,0],[0,0.02]], n_per_cluster)
xy2 = rng.multivariate_normal([-2,2], [[0.2,0],[0,0.2]], n_per_cluster)
xy3 = rng.multivariate_normal([0,-2], [[0.05,0],[0,0.05]], n_per_cluster)

data = np.concatenate((xy1, xy2, xy3), axis=0)
data_labels = np.zeros(3*n_per_cluster, np.uint8)
data_labels[n_per_cluster:2*n_per_cluster] = 1
data_labels[2*n_per_cluster:] = 2

# Apply k-means clustering to fake dataset
centroids, clusters, obj_val = kmeans(data, 3, 5)
plt.figure()
for i in clusters:
    plt.scatter(clusters[i][:,0], clusters[i][:,1], label=i)
    plt.scatter(centroids[:,0], centroids[:,1])
    plt.legend()
plt.savefig("./plots/test_kmeans.png")
