from parse_dataset import *
from kmeans_clustering import *

__PLOT_INDIVIDUALS__ = False
__PLOT_KMEANS__ = True

K = 3
MAX_ITERATIONS = 5

participants = parseDataset('../../EC601/datasets/DR1IFF_J_2017_2018.XPT') # 2017 - 2018 First day
allIronIntakes = list(participants.values())
centroids, clusters, objectives = kmeans(allIronIntakes, K, MAX_ITERATIONS)

if __PLOT_INDIVIDUALS__:
    for seqn, ironIntake in participants.items():
        fileName = "./plots/" + str(seqn) + ".png"
        plotIronIntake("Participant " + str(seqn), ironIntake, fileName)

if __PLOT_KMEANS__:
    for i, centroid in enumerate(centroids):
        fileName = "./plots/centroid_" + str(i) + ".png" 
        plotIronIntake("Centroid " + str(i), centroid, fileName)
    


