# -*- coding: utf-8 -*-
import pandas as pd

import datasplit as ds

import numpy as np

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


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

