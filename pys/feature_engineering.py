# -*- coding: utf-8 -*-
import pandas as pd

import datasplit as ds

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'


def add_cate_features(train, submit):
    train_cat = pd.read_csv(INPUT_PATH + "train_cate_id.csv")
    submit_cat = pd.read_csv(INPUT_PATH + "submit_cate_id.csv")
    col_num = 3
    name_basic = 'cate_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))
    #all_cols = cols_svd_name + ['cat_counts']
    all_cols = cols_svd_name 
    train = train.join(train_cat[all_cols])
    submit = submit.join(submit_cat[all_cols])
    return train, submit

def add_param_features(train, submit):
    train_param = pd.read_csv(INPUT_PATH + "train_param.csv")
    submit_param = pd.read_csv(INPUT_PATH + "submit_param.csv")
    col_num = 3
    name_basic = 'pid_param_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))

    train = train.join(train_param[cols_svd_name])
    submit = submit.join(submit_param[cols_svd_name])
    return train, submit

def add_dow_features(train, submit):
    train_param = pd.read_csv(INPUT_PATH + "train_day_of_week.csv")
    submit_param = pd.read_csv(INPUT_PATH + "submit_day_of_week.csv")
    col_num = 3
    name_basic = 'dow_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))

    cols_extra_name = ["workdays", "weekends"]

    train = train.join(train_param[cols_svd_name + cols_extra_name])
    submit = submit.join(submit_param[cols_svd_name + cols_extra_name])
    return train, submit

def add_svd_features(train, submit):
    train_param = pd.read_csv(INPUT_PATH + "train_svd.csv")
    submit_param = pd.read_csv(INPUT_PATH + "submit_svd.csv")
    col_num = 2
    name_basic = 'all_feature_svd_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))

    train = train.join(train_param[cols_svd_name ])
    submit = submit.join(submit_param[cols_svd_name])
    return train, submit

cat_features = ['age','sex','active_month', 'active_year', 'active_day_of_week', 'limit_get_promoted','limit_get_promoted_ever']

num_features_row = ['limit', 'consume_count', 'consume_amount', 'loan_amount', 'loan_count','plannum','click_count', 'actived_months']


 # num_features_all = ['limit', 'consume_count', 'consume_amount', 'loan_amount', 'loan_count','plannum','click_count',
 # 'consume_count_cum', 'consume_amount_cum', 'loan_amount_cum', 'loan_count_cum', 'click_count_cum',
 # 'plannum_cum', 'actived_months', 'avg_consume_amount_cum', 'median_consume_amount_cum',
 # 'dev_consume_amount_cum',
 # 'dev_median_consume_amount_cum',
 # 'avg_loan_amount_cum',
 # 'dev_loan_amount_cum',
 # 'avg_loan_amount',
 # 'dev_loan_amount',
 # 'avg_click_count_cum',
 # 'median_click_count_cum',
 # 'dev_click_count_cum',
 # 'dev_median_click_count_cum',
 # 'avg_click_count',
 # 'median_click_count',
 # 'dev_click_count',
 # 'dev_median_click_count',
 # 'cate_0',
 # 'cate_1',
 # 'cate_2',
 # 'pid_param_0',
 # 'pid_param_1',
 # 'pid_param_2',
 # 'dow_0',
 # 'dow_1',
 # 'dow_2',
 # 'workdays',
 # 'weekends']

