from parse_dataset import *
from kmeans_clustering import *

__PLOT__ = False

K = 3
MAX_ITERATIONS = 5

participants = parseDataset('../../EC601/datasets/DR1IFF_J_2017_2018.XPT') # 2017 - 2018 First day
allIronIntakes = list(participants.values())
kmeans(allIronIntakes, K, MAX_ITERATIONS)

if __PLOT__:
    for seqn, ironIntake in participants.items():
        fileName = "./plots/" + str(seqn) + ".png"
        plotIndivIronIntake(seqn, ironIntake, fileName)
    


