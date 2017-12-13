# -*- coding: utf-8 -*-
import pandas as pd
import datasplit as ds
from sklearn.decomposition import TruncatedSVD

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

def load_data():
    train = pd.read_csv(INPUT_PATH + 'trainv1.csv')
    submit = pd.read_csv(INPUT_PATH + 'submitv1.csv')
    return train, submit

if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    col_num = 2
    svd = TruncatedSVD(col_num)
    name_basic = 'all_feature_svd_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))

    cum= ['consume_counts_sum','consume_amounts_sum','click_counts_sum']
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

    frames_f1 = [train, submit]
    all_data_uid = pd.concat(frames_f1)
    all_data_uid.reset_index(drop=True, inplace=True)

    # remove uid, try
    train = train.drop(features_to_remove+['target'], axis=1)
    submit = submit.drop(features_to_remove, axis=1)

    train_count = len(train)
    frames = [train, submit]
    all_data = pd.concat(frames)
    all_data.reset_index(drop=True, inplace=True)

    for c in all_data.columns:
        all_data[c] = all_data[c].fillna(0)

    new_data = svd.fit_transform(all_data)
    new_df = pd.DataFrame(new_data, columns=cols_svd_name)

    res = all_data_uid.join(new_df)
    cols = ["uid"] + cols_svd_name
    res = res[cols]
    train_new = res.iloc[0:train_count]
    sub_new = res.iloc[train_count::len(res)]
    print('to csv ........')
    train_new.to_csv(INPUT_PATH +"train_svd.csv", index=False)
    sub_new.to_csv(INPUT_PATH +"submit_svd.csv", index=False)
