from parse_dataset import *
from kmeans_clustering import *
#from tslearn.clustering import TimeSeriesKMeans

__PLOT_INDIVIDUALS__ = False
__PLOT_KMEANS__ = True

K = 3
MAX_ITERATIONS = 5

participants = parseDataset('../../DR1IFF_J_2017_2018.XPT') # 2017 - 2018 First day
allIronRatios = processDataset(participants)
#print(allIronRatios)
'''
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42)
model.fit(allIronRatios)
cluster_labels = model.predict(allIronRatios)
print(cluster_labels, cluster_labels.shape)
'''
centroids, clusters, objectives = kmeans(allIronRatios, K, MAX_ITERATIONS)


if __PLOT_INDIVIDUALS__:
    for seqn, ironIntake in participants.items():
        fileName = "./plots/" + str(seqn) + ".png"
        plotIronIntake("Participant " + str(seqn), ironIntake, fileName)

if __PLOT_KMEANS__:
    for i, centroid in enumerate(centroids):
        fileName = "./plots/centroid_" + str(i) + ".png" 
        plotIronIntake("Centroid " + str(i), centroid, fileName)
    


