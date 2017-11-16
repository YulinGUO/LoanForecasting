# -*- coding: utf-8 -*-
import re

import pandas as pd


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'
CC = '{}_comsume_count'
CM = '{}_consume_amount'
LM = '{}_loan_amount'
LC = '{}_loan_count'
CKC = '{}_click_count'
CUM = '_cum'
basic = ['uid', 'age', 'sex', 'active_date', 'limit']
TARGET = 'target'


def load_data():
    user_info = pd.read_csv(OUTPUT_PATH + 'user_info.csv')
    for c in user_info.columns:
        user_info[c] = user_info[c].fillna(0)
    # cummulated data
    for month in [8, 9, 10, 11, 12]:
        user_info = cummulate_data_by_month(user_info, month)

    for month in [8, 9, 10]:
        lm = LM.format(month)
        lc = LC.format(month)
        user_info[lm] = user_info[lm] - user_info[lc]

    df8 = get_df_by_month(user_info, '8')
    df8 = add_target_by_month(df8, 8, user_info)
    df8 = df8.rename(columns=remove_month_rename)
    df9 = get_df_by_month(user_info, '9')
    df9 = add_target_by_month(df9, 9, user_info)
    df9 = df9.rename(columns=remove_month_rename)
    df10 = get_df_by_month(user_info, '10')
    df10 = add_target_by_month(df10, 10, user_info)
    df10 = df10.rename(columns=remove_month_rename)
    df11 = get_df_by_month(user_info, '11')
    df11 = df11.rename(columns=remove_month_rename)

    frames = [df8, df9, df10]
    result = pd.concat(frames)
    return result, df11


def cummulate_data_by_month(df, month):
    cc_bef = CC.format(month - 1) + CUM
    cm_bef = CM.format(month - 1) + CUM
    lm_bef = LM.format(month - 1) + CUM
    lc_bef = LC.format(month - 1) + CUM
    ckc_bef = CKC.format(month - 1) + CUM

    cc_cur = CC.format(month - 1)
    cm_cur = CM.format(month - 1)
    lm_cur = LM.format(month - 1)
    lc_cur = LC.format(month - 1)
    ckc_cur = CKC.format(month - 1)

    cc_cum = CC.format(month) + CUM
    cm_cum = CM.format(month) + CUM
    lm_cum = LM.format(month) + CUM
    lc_cum = LC.format(month) + CUM
    ckc_cum = CKC.format(month) + CUM

    if month == 8:
        df[cc_cum] = 0
        df[cm_cum] = 0
        df[lm_cum] = 0
        df[lc_cum] = 0
        df[ckc_cum] = 0
    else:
        df[cc_cum] = df[cc_cur] + df[cc_bef]
        df[cm_cum] = df[cm_cur] + df[cm_bef]
        df[lm_cum] = df[lm_cur] + df[lm_bef]
        df[lc_cum] = df[lc_cur] + df[lc_bef]
        df[ckc_cum] = df[ckc_cur] + df[ckc_bef]

    return df


def add_target_by_month(df, month, user_info):
    lm_next = LM.format(month + 1)
    df[TARGET] = user_info[lm_next]
    return df


def get_column_by_month(df, month):
    return list(df.columns[df.columns.str.startswith(month)])


def get_df_by_month(df, month):
    cols = get_column_by_month(df, month)
    basic = ['uid', 'age', 'sex', 'active_date', 'limit']
    all_cols = basic +cols 
    return df[all_cols]


def remove_month_rename(col_name):
    if re.match(r"(\d{1,2}_)(.+)", col_name):
        m = re.search(r"(\d{1,2}_)(.+)", col_name)
        return m.group(2)
    else:
        return col_name


if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    print('to csv ........')
    train.to_csv(INPUT_PATH +"trainv1.csv", index=False)
    submit.to_csv(INPUT_PATH +"submitv1.csv", index=False)

