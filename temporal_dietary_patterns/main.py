from parse_dataset import *

participants = parseDataset('../../EC601/datasets/DR1IFF_J_2017_2018.XPT') # 2017 - 2018 First day
for seqn, ironIntake in participants.items():
    fileName = "./plots/" + str(seqn) + ".png"
    plotIndivIronIntake(seqn, ironIntake, fileName)


