# -*- coding: utf-8 -*-
"""Return ."""
import re

import pandas as pd


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'
CC = '{}_comsume_count'
CM = '{}_consume_amount'
LM = '{}_loan_amount'
LC = '{}_loan_count'
CKC = '{}_click_count'
PM = '{}_plannum'
CUM = '_cum'
basic = ['uid', 'age', 'sex', 'active_date', 'limit']
ACTIVED_MONTHS = "{}_actived_months"
ACTIVE_MONTH = "active_month"
ACTIVE_YEAR = "active_year"
TARGET = 'target'
GPTM = "{}_limit_get_promoted"
GPE = "{}_limit_get_promoted_ever"


def load_data():
    """Return ."""
    user_info = pd.read_csv(OUTPUT_PATH + 'user_info.csv')
    for c in user_info.columns:
        user_info[c] = user_info[c].fillna(0)

    for month in [8, 9, 10]:
        lm = LM.format(month)
        lc = LC.format(month)
        user_info[lm] = user_info[lm] - user_info[lc]
    # cummulated data
    for month in [8, 9, 10, 11, 12]:
        user_info = cummulate_data_by_month(user_info, month)

    user_info = set_actived_month_num(user_info)
    df8 = get_df_by_month(user_info, '8')
    df8 = rename_12_sum(df8)
    # to modify
    # df8[CC.format(8) + CUM] = df8['comsume_counts_sum'] / 4
    # df8[CM.format(8) + CUM] = df8['comsume_amounts_sum'] / 4
    # df8[CKC.format(8) + CUM] = df8['click_counts_sum'] / 4
    
    df8[CC.format(8) + CUM] = (user_info[CC.format(8)] + user_info[CC.format(10)] +  user_info[CC.format(11)])/ 3
    df8[CM.format(8) + CUM] = (user_info[CM.format(8)] + user_info[CM.format(10)] + user_info[CM.format(11)]) / 3
    df8[CKC.format(8) + CUM] = (user_info[CKC.format(8)] + user_info[CKC.format(10)] + user_info[CKC.format(11)]) / 3
    df8[LM.format(8) + CUM] = (user_info[LM.format(8)] + user_info[LM.format(10)] +  user_info[LM.format(11)])/ 3
    df8[LC.format(8) + CUM] = (user_info[LC.format(8)] + user_info[LC.format(10)] + user_info[LC.format(11)]) / 3
    df8[PM.format(8) + CUM] = (user_info[PM.format(8)] + user_info[PM.format(10)] + user_info[PM.format(11)]) / 3

    df8 = add_target_by_month(df8, 8, user_info)
    df8 = df8.rename(columns=remove_month_rename)
    df8 = add_devs(df8)

    df9 = get_df_by_month(user_info, '9')
    df9 = rename_12_sum(df9)
    df9 = add_target_by_month(df9, 9, user_info)
    df9 = df9.rename(columns=remove_month_rename)
    df9 = add_devs(df9)

    df10 = get_df_by_month(user_info, '10')
    df10 = rename_12_sum(df10)
    df10 = add_target_by_month(df10, 10, user_info)
    df10 = df10.rename(columns=remove_month_rename)
    df10 = add_devs(df10)

    df11 = get_df_by_month(user_info, '11')
    df11 = rename_12_sum(df11)
    df11 = df11.rename(columns=remove_month_rename)
    df11 = add_devs(df11)

    frames = [df8, df9, df10]
    # frames = [df9, df10]
    result = pd.concat(frames)
    return result, df11


def cummulate_data_by_month(df, month):
    """Return ."""
    cc_bef = CC.format(month - 1) + CUM
    cm_bef = CM.format(month - 1) + CUM
    lm_bef = LM.format(month - 1) + CUM
    lc_bef = LC.format(month - 1) + CUM
    ckc_bef = CKC.format(month - 1) + CUM
    pm_bef = PM.format(month - 1) + CUM

    cc_cur = CC.format(month - 1)
    cm_cur = CM.format(month - 1)
    lm_cur = LM.format(month - 1)
    lc_cur = LC.format(month - 1)
    ckc_cur = CKC.format(month - 1)
    pm_cur = PM.format(month - 1)

    cc_cum = CC.format(month) + CUM
    cm_cum = CM.format(month) + CUM
    lm_cum = LM.format(month) + CUM
    lc_cum = LC.format(month) + CUM
    ckc_cum = CKC.format(month) + CUM
    pm_cum = PM.format(month) + CUM

    gptm = GPTM.format(month)
    df[gptm] = 0
    lmtm = LM.format(month)
    if month < 12:
        df.loc[df[lmtm] > df['limit'], gptm] = 1
    gpe = GPE.format(month)
    gpe_bef = GPE.format(month - 1)

    df[gpe] = 0
    if month == 8:
        df[cc_cum] = 0
        df[cm_cum] = 0
        df[lm_cum] = 0
        df[lc_cum] = 0
        df[ckc_cum] = 0
        df[pm_cum] = 0
        df[gpe] = df[gptm]
    else:
        df[cc_cum] = df[cc_cur] + df[cc_bef]
        df[cm_cum] = df[cm_cur] + df[cm_bef]
        df[lm_cum] = df[lm_cur] + df[lm_bef]
        df[lc_cum] = df[lc_cur] + df[lc_bef]
        df[ckc_cum] = df[ckc_cur] + df[ckc_bef]
        df[pm_cum] = df[pm_cur] + df[pm_bef]
        df.loc[df[gptm] <= df[gpe_bef], gpe] = df[gpe_bef]

    return df


def add_target_by_month(df, month, user_info):
    """Return ."""
    lm_next = LM.format(month + 1)
    df[TARGET] = user_info[lm_next]
    return df


def get_column_by_month(df, month):
    """Return ."""
    return list(df.columns[df.columns.str.startswith(month)])


def get_column_by_cum(df):
    """Return ."""
    return list(df.columns[df.columns.str.endswith('cum')])


def get_df_by_month(df, month):
    """Return ."""
    cols = get_column_by_month(df, month)
    basic = ['uid', 'age', 'sex', 'active_date', 'limit']
    dates_col = [ACTIVE_MONTH, ACTIVE_YEAR]
    # this get sum of loans of 8,9,10,11
    cols_12_cum = get_column_by_month(df, '12')
    cols_12_cum.remove('12_loan_amount_cum')
    cols_12_cum.remove('12_loan_count_cum')
    cols_12_cum.remove('12_plannum_cum')
    cols_12_cum.remove('12_limit_get_promoted')
    cols_12_cum.remove('12_limit_get_promoted_ever')

    all_cols = basic + dates_col + cols + cols_12_cum
    return df[all_cols]


def remove_month_rename(col_name):
    """Return ."""
    if re.match(r"(\d{1,2}_)(.+)", col_name):
        m = re.search(r"(\d{1,2}_)(.+)", col_name)
        return m.group(2)
    else:
        return col_name


def rename_12_sum(df):
    """Return ."""
    df = df.rename(
     columns={'12_comsume_count_cum': "comsume_counts_sum",
     '12_consume_amount_cum': "comsume_amounts_sum",
#     '12_loan_amount_cum': "loan_amounts_sum",
     # '12_loan_count_cum': "loan_counts_sum",
     '12_click_count_cum': "click_counts_sum"})
    return df


def set_actived_month_num(user_info):
    """Return ."""
    user_info['Date'] = pd.to_datetime(user_info['active_date'], errors='coerce')
    user_info[ACTIVE_MONTH] = user_info['Date'].dt.month
    user_info[ACTIVE_YEAR] = user_info['Date'].dt.year - 2015

    for month in [8, 9, 10, 11]:
        acm = ACTIVED_MONTHS.format(month)
        user_info[acm] = (1 - (user_info['Date'].dt.year- 2015))* 12 + (month - user_info['Date'].dt.month )

    user_info = user_info.drop(['Date'], axis=1)
    return user_info

# median(N1)_by(C1)  \\ 中位数
# mean(N1)_by(C1)  \\ 算术平均数
# mode(N1)_by(C1)  \\ 众数
# min(N1)_by(C1)  \\ 最小值
# max(N1)_by(C1)  \\ 最大值
# std(N1)_by(C1)  \\ 标准差
# var(N1)_by(C1)  \\ 方差
# freq(C2)_by(C1)  \\ 频数

# freq(C1)

def add_devs(df):
    """Return ."""
    # arr = ['comsume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','comsume_amounts_sum']
    arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_count']
    df['cat_age_sex'] = df[['age', 'sex']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_{}'
    DEV_ITEM = 'dev_{}'
    MEDIAN_ITEM = 'median_{}'
    DMEDIAN_ITEM = 'dev_median_{}'
    MODE_ITEM = 'mode_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)


if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    print('to csv ........')
    train.to_csv(INPUT_PATH + "trainv1.csv", index=False)
    submit.to_csv(INPUT_PATH + "submitv1.csv", index=False)
