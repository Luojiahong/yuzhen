# coding=utf-8

import obspy
import numpy as np
import pandas as pd
import csv

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import trigger_onset
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import delayed_sta_lta

from preprocessing import *


# original testing data
def windowedTestingData(sac, point, preSize=10, postSize=20):
    pre = sac.data[point+1-preSize: point+1]
    post = sac.data[point+1: point+1+postSize]
    testingSeq = np.concatenate([pre, post])
    return testingSeq


# MINMAX testing data
def windowedMinMaxTestingData(sac, point, preSize=10, postSize=20):
    pre = sac.data[point+1-preSize: point+1]
    post = sac.data[point+1: point+1+postSize]
    testingSeq = np.concatenate([pre, post])
    if len(testingSeq)!= 0:
        testingSeq = minMaxScale(testingSeq, range=(-100, 100))
    return np.array(testingSeq)


# Inputa a sac and get the on_of array.
def getTrigger(sac, short=10, long=1000):
    df = sac.stats.sampling_rate
    # get cft
    cft = recursive_sta_lta(sac.data, int(short * df), int(long * df))
    # set threshold
    threshold = np.mean(cft) + (np.max(cft) - np.mean(cft))/4
    # get on
    on_of = trigger_onset(cft, threshold, threshold)
    if len(on_of) != 0:
        return on_of[:, 0]
    else:
        return np.array([])


def trainNN():
    # POSITIVE training data
    posPX, posSX = getAllWindowedMinMaxPositiveTrainingData('./sample/example30', preSize=50, postSize=50)
    posPY = np.array([[1]] * len(posPX))
    posSY = np.array([[1]] * len(posSX))

    # NEGATIVE training data
    negX = getSomeWindowedMinMaxNegativeTrainingData('./sample/example30/', size=100, num=200)
    negY = np.array([[0]] * 200)

    # ALL training data
    X = np.concatenate([posPX, posSX, negX])
    Y = np.concatenate([posPY, posSY, negY])

    # 使用keras创建神经网络
    # Sequential是指一层层堆叠的神经网络
    # Dense是指全连接层
    # 定义model
    model = Sequential()
    model.add(Dense(50, input_dim=100, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X, Y, epochs=200, batch_size=32)
    model.save('model.h5')
    return model


def predictOneSacSaved(sacDir):
    res = []
    sac = readOneSac(sacDir)
    ti = sac.stats.starttime
    ti_unix = float(ti.strftime("%s.%f"))

    triggers = getTrigger(sac)
    if len(triggers) != 0:
        i = 1
        for point in triggers:
            testingSeq = windowedMinMaxTestingData(sac, point, preSize=50, postSize=50)
            prob = model.predict(testingSeq.reshape(1, -1))
            if prob > 0.6:
                time = point/100
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix+ 8*3600+time).strftime('%Y%m%d%H%M%S.%f'))
                if i % 2 == 1:
                    wave_type = 'P'
                else:
                    wave_type = 'S'
                res.append([sac.stats.station, time_submission, wave_type])
                i += 1
    return res


if __name__ == '__main__':

    starttime = datetime.datetime.now()
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['Station', 'Time', 'Type'])

    model = trainNN()
    dirs = ['./preliminary/before', './preliminary/after']
    for dir in dirs:
        pathDirBefore = os.listdir(dir)
        for eachFile in pathDirBefore:
            eachFile = os.path.join('%s/%s' % (dir, eachFile))
            print eachFile
            res_one = predictOneSacSaved(eachFile)
            writer.writerows(res_one)

    csvfile.close()

    endtime = datetime.datetime.now()
    print "Spend time:"
    print endtime - starttime


