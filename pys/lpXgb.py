# -*- coding: utf-8 -*-
import math

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

import xgboost as xgb

import feature_engineering as fe


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'


def load_data():
    train = pd.read_csv(INPUT_PATH + 'trainv1.csv')
    train['target'] = train['target'].map(lambda x: data_log(x))
    submit = pd.read_csv(INPUT_PATH + 'submitv1.csv')
    return train, submit


def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}


def merge_dic(dicts):
    ret = {}
    for dict in dicts:
        for key in dict:
            val = dict[key]
            ret[key] = ret[key]+val if key in ret else val
    return ret

def data_log(x):
    if x <= 0:
        return 0
    else:
        return np.math.log(x + 1, 5)

if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    train, submit = fe.add_cate_features(train, submit)
    train, submit = fe.add_param_features(train, submit)
    # train, submit = fe.add_dow_features(train, submit)

    kfold = 10
    skf = KFold(n_splits=kfold,shuffle=True, random_state=42)

    params = {
        'objective': 'reg:linear',
        'max_depth': 6,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'eta': 0.025,
        'gamma': 3,
        'reg_alpha': 0.5,
        'reg_lambda': 1.3,
        'eval_metric': 'rmse',
        'min_child_weight': 6,
        'silent': 1,
        'nthread': 6,
        'seed': 0
    }

    cum= ['consume_counts_sum','consume_amounts_sum','click_counts_sum']
    #actives = [ 'limit_get_promoted','limit_get_promoted_ever']
    #actives = ['dev_consume_count', 'dev_consume_amount', 'dev_loan_amount', 'dev_loan_count', 'dev_plannum', 'dev_click_count']
    #cum= []
    actives = ['dev_median_loan_amount_cum','median_loan_amount_cum','dev_median_loan_amount','median_loan_amount'
    ,'median_three_loan_amount','dev_three_median_loan_amount','dev_three_median_limit']
    features_to_remove = ['uid', 'active_date' ]+cum+actives
#     features_to_remove = ['active_date' ]+cum+actives
    all_features = [x for x in train.columns if not x in features_to_remove+['target']]

    std_zero_cols = []
    for one in all_features:
        if train[one].std() ==0:
            std_zero_cols.append(one)

    features_to_remove = features_to_remove + std_zero_cols
    all_features = [x for x in train.columns if not x in features_to_remove+['target']]

# remove uid, try
    X = train.drop(features_to_remove+['target'], axis=1).values
    y = train.target.values

    sub_id = submit.uid.values
    to_submit = submit.drop(features_to_remove, axis=1)

    sub = pd.DataFrame()
    sub['uid'] = sub_id
    sub['target'] = np.zeros_like(sub_id)

    scores = []

    importances = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        d_submit = xgb.DMatrix(to_submit.values)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        mdl = xgb.train(params, d_train, 4000, watchlist, early_stopping_rounds = 70, verbose_eval = 50)

        f_importance = {}
        try:
            f_importance = get_xgb_imp(mdl, all_features)
        except:
            print("error")
        print(f_importance)
        importances.append(f_importance)

        valid_pred = mdl.predict(d_valid, ntree_limit = mdl.best_ntree_limit)

        score_this = np.sqrt(mean_squared_error(valid_pred, y_valid))
        print(score_this)
        scores.append(score_this)

        print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
        # Predict on our test data
        p_test = mdl.predict(d_submit, ntree_limit=mdl.best_ntree_limit)

        sub['target'] += p_test / kfold

    imp_sum = merge_dic(importances)
    sort_rec = sorted(imp_sum.items(), key=lambda x:x[1])
    print(sort_rec)

    print('cv avg scores %s' % np.mean(scores))
    from datetime import datetime
    sub.loc[sub.target < 0, 'target'] = 0
    sub.to_csv(OUTPUT_PATH +"subByXgb{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
