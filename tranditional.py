from obspy import read
import os
import csv
from NN import *
from preprocessing import *
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import trigger_onset
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import delayed_sta_lta


if __name__ == '__main__':

    starttime = datetime.datetime.now()
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['Station', 'Time', 'Type'])

    dirs = ['./preliminary/before', './preliminary/after']
    for dir in dirs:
        pathDirBefore = os.listdir(dir)
        for eachFile in pathDirBefore:

            eachFile = os.path.join('%s/%s' % (dir, eachFile))
            print eachFile
            sac = readOneSac(eachFile)
            ti = sac.stats.starttime
            ti_unix = float(ti.strftime("%s.%f"))

            triggers = getTrigger(sac)
            if len(triggers) != 0:
                i = 1
                for triger in triggers:
                    res = []
                    time = triger / 100
                    time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                    if i % 2 == 1:
                        wave_type = 'P'
                    else:
                        wave_type = 'S'
                    i += 1
                    res.append([sac.stats.station, time_submission, wave_type])
                    writer.writerows(res)

    csvfile.close()

    endtime = datetime.datetime.now()
    print "Spend time:"
    print endtime - starttime



# test = np.array([1,1,1,1,1,1,1,1,1,10,40,50,100,200,300,400,60,30,200,400,40,50,100,200,300,400,60,30,200,400])
# test1 = np.array([1,1,1,1,1,1,1,1,10,10,40,50,100,200,300,400,60,30,200,400,40,50,100,200,300,400,60,30,200,400])
# test2 = np.array([30,200,400,40,50,100,200,300,400,60,30,200,400,1,1,1,1,1,1,1,1,1,10,40,50,100,200,300,400,60])
#
# a = minMaxScale(test, range=(-100,100))
# b = minMaxScale(test1, range=(-100,100))
# c = minMaxScale(test2, range=(-100,100))
#
# print model.predict(a.reshape(1, -1))
# print model.predict(b.reshape(1, -1))
# print model.predict(c.reshape(1, -1))
#
# print '-------P[0 1]--------'
# print model.predict(X[0].reshape(1, -1))
# print model.predict(X[1].reshape(1, -1))
# print model.predict(X[2].reshape(1, -1))
# print model.predict(X[3].reshape(1, -1))
# print model.predict(X[4].reshape(1, -1))
# print '-------S[0 1]--------'
# print model.predict(X[100].reshape(1, -1))
# print model.predict(X[101].reshape(1, -1))
# print model.predict(X[102].reshape(1, -1))
# print model.predict(X[103].reshape(1, -1))
# print model.predict(X[104].reshape(1, -1))
# print '-------N[1 0]--------'
# print model.predict(X[200].reshape(1, -1))
# print model.predict(X[201].reshape(1, -1))
# print model.predict(X[202].reshape(1, -1))
# print model.predict(X[203].reshape(1, -1))
# print model.predict(X[204].reshape(1, -1))
