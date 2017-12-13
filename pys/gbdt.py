# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import feature_engineering as fe
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.cross_validation import KFold 

input = "../input/"
file_path = "../output/"

def data_log(x):
    if x <= 0:
        return 0
    else:
        return np.math.log(x+1, 5)

def RMSE_loss_fun(ground_truth, predictions):
    return np.sqrt(mean_squared_error(ground_truth, predictions))

def load_data():
    data = pd.read_csv(input + "trainv1.csv")
    submit = pd.read_csv(input + "submitv1.csv")

    data, submit = fe.add_cate_features(data, submit)
    data, submit = fe.add_param_features(data, submit)
    data, submit = fe.add_dow_features(data, submit)

    for col in data.columns:
        if col.endswith("sum"):
            data = data.drop(col, axis=1)
    for col in submit.columns:
        if col.endswith("sum"):
            submit = submit.drop(col, axis=1)
            
    for c in data.columns:
        data[c] = data[c].fillna(0)  
    for c in submit.columns:
        submit[c] = submit[c].fillna(0)

    return data,submit

def model_train(x,y):
    kf = KFold(len(y), 10)
    model_list = []
    score_list = []
    for train, test in kf:
    #     print train, test
        x_train, x_test = x.ix[train], x.ix[test]
        y_train, y_test = y[train], y[test]
        gbdt = GradientBoostingRegressor(n_estimators=180,learning_rate=0.08,subsample=1,loss="ls",
                                         alpha=0.9,max_features=None,max_depth=5,min_samples_split=5,min_samples_leaf=5)
        gbdt.fit(x_train, y_train)
        model_list.append(gbdt)
        y_pred = gbdt.predict(x_test)
        rmse_gbdt = RMSE_loss_fun(y_test, y_pred)
        score_list.append(rmse_gbdt)
        print(rmse_gbdt)
    print("avg scores:", round(np.mean(score_list), 6)) 
    return model_list

def predict2csv(model_list, submit):
    dt = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
    submit_train = submit.drop(["uid", "active_date"], axis = 1)
    submit_uid = submit[["uid"]]
    submit_sum = 0
    for i in range(len(model_list)):
        submit_sum += pd.DataFrame(model_list[i].predict(submit_train), columns=["target"])
    res = submit_uid.join(submit_sum/len(model_list))
    res.to_csv(file_path+"gbdt_submit_withall_rank_"+dt+".csv", index=False)

if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    print('model training ........')
    x = train.drop(["uid", "target", "active_date"], axis=1)
    y = train["target"].map(lambda x:data_log(x))
    model_list = model_train(x, y)
    print("to csv.....")
    predict2csv(model_list, submit)