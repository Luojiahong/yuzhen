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
from obspy.signal.filter import *
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


# Inputa a sac and get the on_of array. pujun 10 1000
def getTrigger(sac, short=2, long=25):
    df = sac.stats.sampling_rate
    print 'sampling_rate = '
    print df
    # get cft
    cft = recursive_sta_lta(sac.data, int(short * df), int(long * df))
    # set threshold
    threshold = np.mean(cft) + (np.max(cft) - np.mean(cft))/4
    if np.isnan(threshold) == 1:
        print 'thre = nan'
        threshold = 3.2
    # get on
    #gk change
    #on_of = trigger_onset(cft, threshold, threshold)
    on_of = trigger_onset(cft, threshold*1.6, threshold*1.1)
    if len(on_of) != 0:
        return on_of[:, 0]
    else:
        return np.array([])


def trainNN(positive_data_type='all', negative_data_num=200):
    # POSITIVE training data
    posPX, posSX = getAllWindowedMinMaxPositiveTrainingData('./sample/example30', preSize=50, postSize=50)
    posPY = np.array([[1]] * len(posPX))
    posSY = np.array([[1]] * len(posSX))

    # NEGATIVE training data
    negX = getSomeWindowedMinMaxNegativeTrainingData('./sample/example30/', size=100, num=negative_data_num)
    negY = np.array([[0]] * len(negX))

    if positive_data_type == 'p' or positive_data_type == 'P':
        # Only P positive training data
        X = np.concatenate([posPX, negX])
        Y = np.concatenate([posPY, negY])
    elif positive_data_type == 's' or positive_data_type == 'S':
        # Only S positive training data
        X = np.concatenate([posSX, negX])
        Y = np.concatenate([posSY, negY])
    else:
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
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X, Y, epochs=200, batch_size=32)
    model.save('model.h5')
    print len(X), len(Y)
    return model


def predictOneSacSaved(sacDir):
    res = []
    sac = readOneSac(sacDir)
    #dai tong lv bo
    tr_filt = sac.copy()
    tr_filt.filter('bandpass', freqmin=1,freqmax=30, corners=4, zerophase=False)
    #t = np.arange(0, sac.stats.npts / sac.stats.sampling_rate, sac.stats.delta)
    #plt.subplot(211)
    #plt.plot(sac, sac.data, 'k')
    #plt.ylabel('Raw Data')
    #plt.subplot(212)
    #plt.plot(t, tr_filt.data, 'k')
    #plt.ylabel('Lowpassed Data')
    #plt.xlabel('Time [s]')
    #plt.suptitle(sac.stats.starttime)
    #plt.show()
    ti = sac.stats.starttime
    ti_unix = float(ti.strftime("%s.%f"))

    triggers = getTrigger(tr_filt)
    if len(triggers) != 0:
        i = 1
        b = 0.1
        for point in triggers:
            
            testingSeq = windowedMinMaxTestingData(tr_filt, point, preSize=50, postSize=50)
            prob = model.predict(testingSeq.reshape(1, -1))
            if prob > 0.6:
                time = round(float (point) / 100.00,2) #round(a/b,2)
                #print point,time
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix+ 8*3600+time).strftime('%Y%m%d%H%M%S.%f'))
                wave_type = 'P'
                res.append([sac.stats.station, time_submission, wave_type])
               
    return res


if __name__ == '__main__':

    starttime = datetime.datetime.now()
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    #writer.writerow(['Station', 'Time', 'Type'])

    model = trainNN(positive_data_type='all', negative_data_num = 100)
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


