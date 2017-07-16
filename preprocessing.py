from obspy import read
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from obspy.core import UTCDateTime


# Read one sac. Return a trace.
def readOneSac(fileAdd):
    # return one Trace
    singleSac = read(fileAdd, debug_headers=True)
    trace = singleSac[0]
    return trace


# Scale a trace.data.
def minMaxScale(data, range=(-1, 1)):
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=range)
    scaledData = scaler.fit_transform(data).reshape(1, -1)[0]
    return scaledData


# Get the windowed POSITIVE data for training.
def windowedTrainData(sac, preSize=10, postSize=20):
    relativeStartTime = sac.stats.sac.b
    sampling_rate = sac.stats.sampling_rate

    # P-wave windowed data for training
    PTime = sac.stats.sac.a - relativeStartTime
    nP = int(PTime * sampling_rate)
    preP = sac.data[nP + 1 - preSize: nP + 1]
    postP = sac.data[nP + 1: nP + 1 + postSize]
    # # If not enough
    # if len(preP) != preSize:
    #     preP += preP[-1] * (preSize - len(preP))
    # if len(postP) != postSize:
    #     postP += postP[-1] * (postSize - len(postP))
    trainP = np.concatenate([preP, postP])

    # S-wave windowed data for training.
    STime = sac.stats.sac.t0 - relativeStartTime
    nS = int(STime * sampling_rate)
    preS = sac.data[nS + 1 - preSize: nS + 1]
    postS = sac.data[nS + 1: nS + postSize + 1]
    # # If not enough
    # if len(preS) != preSize:
    #     preS += preS[-1] * (preSize - len(preS))
    # if len(postS) != postSize:
    #     postS += postS[-1] * (postSize - len(postS))
    trainS = np.concatenate([preS, postS])

    return np.array(trainP), np.array(trainS)


# Get ALL windowed POSITIVE data for training.
def getAllWindowedPositiveTrainingData(dirAdd, preSize=10, postSize=20):
    trainP = []
    trainS = []
    pathDir = os.listdir(dirAdd)
    for allDir in pathDir:
        eachTrainFile = os.path.join('%s/%s' % (dirAdd, allDir))
        eachSac = readOneSac(eachTrainFile)
        eachTrainP, eachTrainS = windowedTrainData(eachSac, preSize, postSize)
        # In case of null S or P
        if len(eachTrainP) != 0:
            trainP.append(eachTrainP)
        if len(eachTrainS) != 0:
            trainS.append(eachTrainS)
    return np.array(trainP), np.array(trainS)


# Get ALL windowed MINMAX POSITIVE data for training.
def getAllWindowedMinMaxPositiveTrainingData(dirAdd, preSize=10, postSize=20):
    trainP = []
    trainS = []
    pathDir = os.listdir(dirAdd)
    for allDir in pathDir:
        eachTrainFile = os.path.join('%s/%s' % (dirAdd, allDir))
        eachSac = readOneSac(eachTrainFile)
        eachTrainP, eachTrainS = windowedTrainData(eachSac, preSize, postSize)
        # In case of null S or P
        if len(eachTrainP) != 0:
            eachTrainP = minMaxScale(eachTrainP, range=(-100,100))
            trainP.append(eachTrainP)
        if len(eachTrainS) != 0:
            eachTrainS = minMaxScale(eachTrainS, range=(-100, 100))
            trainS.append(eachTrainS)
    return np.array(trainP), np.array(trainS)


# Get ALL windowed STANDARD POSITIVE data for training.
def getAllWindowedStandardPositiveTrainingData(dirAdd, preSize=10, postSize=20):
    trainP = []
    trainS = []
    pathDir = os.listdir(dirAdd)
    for allDir in pathDir:
        eachTrainFile = os.path.join('%s/%s' % (dirAdd, allDir))
        eachSac = readOneSac(eachTrainFile)
        eachTrainP, eachTrainS = windowedTrainData(eachSac, preSize, postSize)
        # In case of null S or P
        if len(eachTrainP) != 0:
            eachStandardTrainP = StandardScaler().fit_transform(eachTrainP.reshape(-1, 1)).reshape(1, -1)[0]
            trainP.append(eachStandardTrainP)
        if len(eachTrainS) != 0:
            eachStandardTrainS = StandardScaler().fit_transform(eachTrainS.reshape(-1, 1)).reshape(1, -1)[0]
            trainS.append(eachStandardTrainS)
    return np.array(trainP), np.array(trainS)


# Get the windowed NEGATIVE data for training.
def getWindowedNegativeTrainingData(sac, size=30):
    dataLen = len(sac.data)
    randStart = np.random.randint(1,dataLen-size)
    trainNegative = sac.data[randStart:randStart+size]
    return np.array(trainNegative)


# Get SOME windowed NEGATIVE data for training.
def getSomeWindowedNegativeTrainingData(dirAdd, size=30, num=100):
    trainNegative=[]
    pathDir = os.listdir(dirAdd)
    numFile = len(pathDir)
    # Randomly select a file to extract NEGATIVE training data
    for i in range(num):
        randomFileDir = pathDir[np.random.randint(1, numFile)]
        randomFile = os.path.join('%s/%s' % (dirAdd, randomFileDir))
        eachSac = readOneSac(randomFile)
        eachNegativeTrainingData = getWindowedNegativeTrainingData(eachSac, size)
        trainNegative.append(eachNegativeTrainingData)
    return np.array(trainNegative)


# Get SOME windowed MINMAX NEGATIVE data for training.
def getSomeWindowedMinMaxNegativeTrainingData(dirAdd, size=30, num=100):
    trainNegative=[]
    pathDir = os.listdir(dirAdd)
    numFile = len(pathDir)
    # Randomly select a file to extract NEGATIVE training data
    for i in range(num):
        randomFileDir = pathDir[np.random.randint(1, numFile)]
        randomFile = os.path.join('%s/%s' % (dirAdd, randomFileDir))
        eachSac = readOneSac(randomFile)
        eachNegativeTrainingData = getWindowedNegativeTrainingData(eachSac, size)
        eachNegativeTrainingData = minMaxScale(eachNegativeTrainingData, range=(-100, 100))
        trainNegative.append(eachNegativeTrainingData)
    return np.array(trainNegative)


# Get SOME windowed STANDARD NEGATIVE data for training.
def getSomeWindowedStandardNegativeTrainingData(dirAdd, size=30, num=100):
    trainStandardNegative=[]
    pathDir = os.listdir(dirAdd)
    numFile = len(pathDir)
    # Randomly select a file to extract NEGATIVE training data
    for i in range(num):
        randomFileDir = pathDir[np.random.randint(1, numFile)]
        randomFile = os.path.join('%s/%s' % (dirAdd, randomFileDir))
        eachSac = readOneSac(randomFile)
        eachNegativeTrainingData = getWindowedNegativeTrainingData(eachSac, size)
        eachStandardNegativeTrainingData = StandardScaler().fit_transform(eachNegativeTrainingData.reshape(-1, 1)).reshape(1, -1)[0]
        trainStandardNegative.append(eachStandardNegativeTrainingData)
    return np.array(trainStandardNegative)


# Plot the training data with P/S arrival time.
# '../sample/example30/01.JMG.BHE.SAC'
def plotPositiveTrainingDataSac(sacAdd):
    sac = readOneSac(sacAdd)
    sampling_rate = sac.stats.sampling_rate
    ax = plt.subplot(111)
    plt.plot(sac.data, 'k')
    ymin, ymax = ax.get_ylim()
    PTime = sac.stats.sac.a - sac.stats.sac.b
    STime = sac.stats.sac.t0 - sac.stats.sac.b
    plt.vlines(PTime * sampling_rate, ymin, ymax, color='r', linewidth=2)
    plt.vlines(STime * sampling_rate, ymin, ymax, color='b', linewidth=2)
    plt.show()


def plotPositiveTrainingDataSegment(segmentData):
    plt.plot(range(len(segmentData)), segmentData)
    plt.show()


# Main
if __name__ == '__main__':
    # sac = readOneSac('../sample/example30/01.JMG.BHE.SAC')
    # a,b = windowedTrainData(sac, 10, 20)
    # print a, len(a)
    # print b, len(b)
    #
    # plotPositiveTrainingDataSac('../sample/example30/16.JMG.BHE.SAC')
    # p, s = getAllWindowedStandardPositiveTrainingData('../sample/example30')
    # print p[0]
    # print s[0]
    # plotPositiveTrainingDataSegment(p[1])

    # sac = readOneSac('../sample/example30/01.JMG.BHE.SAC')
    # a = getSomeWindowedStandardNegativeTrainingData('../sample/example30/', size=30, num=100)
    # print a, a.shape
    # # print StandardScaler().fit_transform(a.reshape(-1,1)).reshape(1,-1)[0].shape
    plotPositiveTrainingDataSac('../sample/example30/01.JMG.BHE.SAC')