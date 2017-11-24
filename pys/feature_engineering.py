# -*- coding: utf-8 -*-
import pandas as pd


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

