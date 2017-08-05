import csv
import datetime

from gNN import getTrigger, var_aic, trainNN, os
from preprocessing import readOneSac
from features import forward_backward_ratio_point


def predict_one_sac_by_stalta_aic_var(trace):
    # read
    sac = readOneSac(trace)
    res = []
    p_tmp = []
    s_tmp = []
    # filter
    tr_filt = sac.copy()
    tr_filt.filter('bandpass', freqmin=8, freqmax=18, corners=4, zerophase=False)
    # time type
    ti = sac.stats.starttime
    ti_unix = float(ti.strftime("%s.%f"))

    # triggers by recursive_sta_lta
    triggers = getTrigger(tr_filt)

    if len(triggers) != 0:
        for point in triggers:
            # aicpoints by aic from triggers
            ratio = forward_backward_ratio_point(sac, point)
            if 0 < ratio < 0.1:
                wave_type = 'P'
                aicpoint = var_aic(tr_filt, point)
                time = round(float(aicpoint) / 100.00, 2)  # round(a/b,2)
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                res.append([sac.stats.station, time_submission, wave_type])
                p_tmp.append(aicpoint)
            elif 0.1 < ratio < 0.2:
                wave_type = 'S'
                aicpoint = var_aic(tr_filt, point)
                time = round(float(aicpoint) / 100.00, 2)  # round(a/b,2)
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                res.append([sac.stats.station, time_submission, wave_type])
                s_tmp.append(aicpoint)
            else:
                wave_type = 'P'
                aicpoint = var_aic(tr_filt, point)
                time = round(float(aicpoint) / 100.00, 2)  # round(a/b,2)
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                res.append([sac.stats.station, time_submission, wave_type])

    return res, p_tmp, s_tmp


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)

    dirs = ['./preliminary/after']
    for dir in dirs:
        pathDirBefore = os.listdir(dir)
        for eachFile in pathDirBefore:
            eachFile = os.path.join('%s/%s' % (dir, eachFile))
            print eachFile
            res_one, p_tmp, s_tmp = predict_one_sac_by_stalta_aic_var(eachFile)
            writer.writerows(res_one)

    csvfile.close()

    endtime = datetime.datetime.now()
    print "Spend time:"
    print endtime - starttime

