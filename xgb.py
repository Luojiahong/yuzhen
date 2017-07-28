# coding=utf-8
import datetime

from features import *
import csv
import pandas as pd
from NN import getTrigger
from sklearn.cross_validation import train_test_split

class xgb_model(object):
    def __init__(self):
        self.params = {'objective': 'multi:softmax',
                       'eta': 0.05,
                       'gamma': 0.1,
                       'max_depth': 6,
                       'silent': 1,
                       'num_class': 3}
        self.num_rounds = 10

    def traditional(self):
        # 获得到时的features
        p, s, other = get_all_positive_features('./example30')

        # 构建p/s/other的X/y
        data_X = np.concatenate([p.values, s.values, other.values])
        data_y = np.concatenate([np.array([[0]] * len(p.values)),
                                 np.array([[1]] * len(s.values)),
                                 np.array([[2]] * len(other.values))])

        # 划分train和test
        train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_y, test_size=0.3, random_state=10)

        # xgboost
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        # #setup parameters for xgboost
        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        params = self.params
        num_rounds = self.num_rounds
        bst = xgb.train(params, xg_train, num_rounds, watchlist)
        # #get prediction
        pred = bst.predict(xg_test)
        print pred
        print ('predicting, classification error=%f' % (
            sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))
        return bst

    def bagging(self, rounds=10):
        # 获得到时的features
        p, s, other = get_all_positive_features('./example30')
        # 构建p/s/other的X/y
        data_X = np.concatenate([p.values, s.values, other.values])
        data_y = np.concatenate([np.array([[0]] * len(p.values)),
                                 np.array([[1]] * len(s.values)),
                                 np.array([[2]] * len(other.values))])
        # 缝合X/y
        data = np.hstack([data_X, data_y])
        # xgb parameters
        params = self.params
        num_rounds = self.num_rounds
        # bagging
        # #存储k个models
        all_pred_res = []
        for i in range(1, rounds + 1):
            # 取80%的数据用于训练，20%用于观察
            np.random.shuffle(data)
            train_data = data[0: int(0.8 * len(data))]
            test_data = data[int(0.8 * len(data)):]
            # 每次训练一个xgb，并存储model
            train_X = train_data[:, :-1]
            train_Y = train_data[:, -1]
            test_X = test_data[:, :-1]
            test_Y = test_data[:, -1]
            xg_train = xgb.DMatrix(train_X, label=train_Y)
            xg_test = xgb.DMatrix(test_X, label=test_Y)
            sub_bst = xgb.train(params, xg_train, num_rounds)
            pred = sub_bst.predict(xg_test)
            print ('model %d: predicting error=%f' % (
            i, sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))
            # 存储子model
            model_path = './model/xgb_cv_model_%d.model' % i
            sub_bst.save_model(model_path)
            # 每个model对全集进行预测
            xg_all_data = xgb.DMatrix(data[:, :-1], label=data[:, -1])
            pred_all = sub_bst.predict(xg_all_data)
            # 保存每个model对全集的预测结果
            all_pred_res.append(pred_all.tolist())
        final_pred_res = pd.DataFrame(all_pred_res).mode().iloc[0].values
        all_Y = data[:, -1]
        print ('final result: predicting error=%f' % (
        sum(int(final_pred_res[i]) != all_Y[i] for i in range(len(all_Y))) / float(len(all_Y))))


def predict_one_trace_saved(trace_dir, bst):
    res = []
    trace = readOneSac(trace_dir)
    ti = trace.stats.starttime
    ti_unix = float(ti.strftime("%s.%f"))

    points = getTrigger(trace)
    if len(points) != 0:
        points_features = get_points_features(trace, points).values
        xg_points_features = xgb.DMatrix(points_features)
        preds = bst.predict(xg_points_features)
        for i in range(0, len(preds)):
            if preds[i] == 0:
                time = round(float(points[i]) / 100.00, 2)
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                wave_type = 'P'
                res.append([trace.stats.station, time_submission, wave_type])
            if preds[i] == 1:
                time = round(float(points[i]) / 100.00, 2)
                time_submission = float(datetime.datetime.fromtimestamp(ti_unix + 8 * 3600 + time).strftime('%Y%m%d%H%M%S.%f'))
                wave_type = 'S'
                res.append([trace.stats.station, time_submission, wave_type])
    return res


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    # writer.writerow(['Station', 'Time', 'Type'])

    bst = xgb_model().traditional()
    dirs = [ './preliminary/after']
    for dir in dirs:
        pathDirBefore = os.listdir(dir)
        for eachFile in pathDirBefore:
            eachFile = os.path.join('%s/%s' % (dir, eachFile))
            print eachFile
            res_one = predict_one_trace_saved(eachFile, bst)
            writer.writerows(res_one)

    csvfile.close()

    endtime = datetime.datetime.now()
    print "Spend time:"
    print endtime - starttime