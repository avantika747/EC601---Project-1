import xport
import math
import matplotlib.pyplot as plt

__DEBUG__ = True

INDIV_FOODS_SEQN_INDEX = 0
INDIV_FOODS_TIME_INDEX = 13
INDIV_FOODS_IRON_INDEX = 55

TOTAL_HOURS = 24
SEC_PER_HR = 3600

def parseDataset(fileName):
    '''
    NHANES Dietary Data - Dietary Interview - Individual Foods
    '''
    
    if __DEBUG__: count = 0
    
    participants = {} # seqn : [iron intake at each hour of the day]
    with open(fileName, 'rb') as f:
        for row in xport.Reader(f):
            seqn = int(row[INDIV_FOODS_SEQN_INDEX])
            time = int(math.floor(row[INDIV_FOODS_TIME_INDEX] / SEC_PER_HR))
            iron = row[INDIV_FOODS_IRON_INDEX]
            if seqn not in participants:
                participants[seqn] = [0] * TOTAL_HOURS
                participants[seqn][time] = iron
            else:
                participants[seqn][time] += iron

            if __DEBUG__:
                count += 1
                if count > 100:
                    break
    return participants

def plotIndivIronIntake(seqn, ironIntake, imgFileName):
    '''
    seqn: participant's unique number
    ironIntake: [iron intake at ith hour] * 24
    imgFileName: name of output image file
    '''

    if __DEBUG__:
        print("Plotting participant " + str(seqn) + "'s iron intake...")
    
    plt.xlim([0, TOTAL_HOURS])
    plt.xticks(range(TOTAL_HOURS), range(TOTAL_HOURS))
    plt.xlabel("Time (hours)")
    plt.ylabel("Iron Intake (mg)")
    plt.title("Participant " + str(seqn))
    for hr, iron in enumerate(ironIntake):
        plt.bar(hr, iron)
    plt.savefig(imgFileName)


