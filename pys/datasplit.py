# -*- coding: utf-8 -*-
"""Return ."""
import re

import pandas as pd

import numpy as np

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'
CC = '{}_consume_count'
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
ACTIVE_DOW = "active_day_of_week"
TARGET = 'target'
GPTM = "{}_limit_get_promoted"
GPE = "{}_limit_get_promoted_ever"
AVG = '_avg'

arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_count', 
'limit','actived_months','lmp_reste','lmp_pay']
#'lm_cum_per_loan_cum','lm_cum_per_plan_cum','lm_per_plan','cm_per_cc'

def load_data():
    """Return ."""
    user_info = pd.read_csv(OUTPUT_PATH + 'user_info.csv')
    lap = pd.read_csv(OUTPUT_PATH +"loan_pay_next.csv")
    #add loan amount pay next month
    user_info = user_info.merge(lap,how='left', on="uid")
    for c in user_info.columns:
        user_info[c] = user_info[c].fillna(0)

    # for month in [8, 9, 10]:
    #     lm = LM.format(month)
    #     lc = LC.format(month)
    #     user_info[lm] = user_info[lm] - user_info[lc]
    # cummulated data
    for month in [8, 9, 10, 11, 12]:
        user_info = cummulate_data_by_month(user_info, month)

    user_info = set_actived_month_num(user_info)
    df8 = get_df_by_month(user_info, '8')
    df8 = rename_12_sum(df8)
    # to modify
    # df8[CC.format(8) + CUM] = df8['consume_counts_sum'] / 4
    # df8[CM.format(8) + CUM] = df8['consume_amounts_sum'] / 4
    # df8[CKC.format(8) + CUM] = df8['click_counts_sum'] / 4
    df8[CC.format(8) + CUM] = (user_info[CC.format(8)] + user_info[CC.format(10)] +  user_info[CC.format(11)])/ 3
    df8[CM.format(8) + CUM] = (user_info[CM.format(8)] + user_info[CM.format(10)] + user_info[CM.format(11)]) / 3
    df8[CKC.format(8) + CUM] = (user_info[CKC.format(8)] + user_info[CKC.format(10)] + user_info[CKC.format(11)]) / 3
    df8[LM.format(8) + CUM] = (user_info[LM.format(8)] + user_info[LM.format(10)] +  user_info[LM.format(11)])/ 3
    df8[LC.format(8) + CUM] = (user_info[LC.format(8)] + user_info[LC.format(10)] + user_info[LC.format(11)]) / 3
    df8[PM.format(8) + CUM] = (user_info[PM.format(8)] + user_info[PM.format(10)] + user_info[PM.format(11)]) / 3

    # df8[CC.format(8) + CUM + AVG] = df8[CC.format(8) + CUM]
    # df8[CM.format(8) + CUM + AVG] = df8[CM.format(8) + CUM]
    # df8[CKC.format(8) + CUM + AVG] = df8[CKC.format(8) + CUM]
    # df8[LM.format(8) + CUM + AVG] = df8[LM.format(8) + CUM]
    # df8[LC.format(8) + CUM + AVG] = df8[LC.format(8) + CUM]
    # df8[PM.format(8) + CUM + AVG] = df8[PM.format(8) + CUM]

    df8 = add_target_by_month(df8, 8, user_info)
    df8 = df8.rename(columns=remove_month_rename)
    df8 = add_combine_features(df8)
    df8 = add_devs(df8)
    df8 = add_devs_with3cat(df8)
    df8 = add_devs_another(df8)
    df8 = add_devs_date(df8)
    #df8 = add_rank_features(df8)

    df9 = get_df_by_month(user_info, '9')
    df9 = rename_12_sum(df9)
    df9 = add_target_by_month(df9, 9, user_info)
    df9 = df9.rename(columns=remove_month_rename)
    df9 = add_combine_features(df9)
    df9 = add_devs(df9)
    df9 = add_devs_with3cat(df9)
    df9 = add_devs_another(df9)
    df9 = add_devs_date(df9)
    #df9 = add_rank_features(df9)

    df10 = get_df_by_month(user_info, '10')
    df10 = rename_12_sum(df10)
    df10 = add_target_by_month(df10, 10, user_info)
    df10 = df10.rename(columns=remove_month_rename)
    df10 = add_combine_features(df10)
    df10 = add_devs(df10)
    df10 = add_devs_with3cat(df10)
    df10 = add_devs_another(df10)
    df10 = add_devs_date(df10)
    #df10 = add_rank_features(df10)

    df11 = get_df_by_month(user_info, '11')
    df11 = rename_12_sum(df11)
    df11 = df11.rename(columns=remove_month_rename)
    df11 = add_combine_features(df11)
    df11 = add_devs(df11)
    df11 = add_devs_with3cat(df11)
    df11 = add_devs_another(df11)
    df11 = add_devs_date(df11)
    #df11 = add_rank_features(df11)

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

    # cc_cum_avg = CC.format(month) + CUM + AVG
    # cm_cum_avg = CM.format(month) + CUM + AVG
    # lm_cum_avg = LM.format(month) + CUM + AVG
    # lc_cum_avg = LC.format(month) + CUM + AVG
    # ckc_cum_avg = CKC.format(month) + CUM + AVG
    # pm_cum_avg = PM.format(month) + CUM + AVG

    # months = month - 8

    # if month <> 12:
    #     if month == 8:
    #         df[cc_cum_avg] = 0
    #         df[cm_cum_avg] = 0
    #         df[lm_cum_avg] = 0
    #         df[lc_cum_avg] = 0
    #         df[ckc_cum_avg] = 0
    #         df[pm_cum_avg] = 0
    #     else:
    #         df[cc_cum_avg] = df[cc_cum] / months
    #         df[cm_cum_avg] = df[cm_cum] / months
    #         df[lm_cum_avg] = df[lm_cum] / months
    #         df[lc_cum_avg] = df[lc_cum] / months
    #         df[ckc_cum_avg] = df[ckc_cum] / months
    #         df[pm_cum_avg] = df[pm_cum] / months

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
    dates_col = [ACTIVE_MONTH, ACTIVE_YEAR, ACTIVE_DOW]
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
     columns={'12_consume_count_cum': "consume_counts_sum",
     '12_consume_amount_cum': "consume_amounts_sum",
#     '12_loan_amount_cum': "loan_amounts_sum",
     # '12_loan_count_cum': "loan_counts_sum",
     '12_click_count_cum': "click_counts_sum"})
    return df


def set_actived_month_num(user_info):
    """Return ."""
    user_info['Date'] = pd.to_datetime(user_info['active_date'], errors='coerce')
    user_info[ACTIVE_MONTH] = user_info['Date'].dt.month
    user_info[ACTIVE_YEAR] = user_info['Date'].dt.year - 2015
    user_info[ACTIVE_DOW] = user_info['Date'].dt.dayofweek

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
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']

    df['cat_age_sex'] = df[['age', 'sex']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_{}'
    DEV_ITEM = 'dev_{}'
    MEDIAN_ITEM = 'median_{}'
    DMEDIAN_ITEM = 'dev_median_{}'
    MODE_ITEM = 'mode_{}'
    MAX_ITEM = 'max_{}'
    MIN_ITEM = 'min_{}'
    DMAX_ITEM = 'dev_max_{}'
    DMIN_ITEM = 'dev_min_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)

def add_devs_another(df):
    """Return ."""
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']
    df['cat_age_sex'] = df[['sex', 'limit']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_sl_{}'
    DEV_ITEM = 'dev_sl_{}'
    MEDIAN_ITEM = 'median_sl_{}'
    DMEDIAN_ITEM = 'dev_sl_median_{}'
    MODE_ITEM = 'mode_sl_{}'
    MAX_ITEM = 'max_sl_{}'
    MIN_ITEM = 'min_sl_{}'
    DMAX_ITEM = 'dev_max_sl_{}'
    DMIN_ITEM = 'dev_min_sl_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)

def add_devs_third(df):
    """Return ."""
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']
    df['cat_age_sex'] = df[['age', 'limit']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_al_{}'
    DEV_ITEM = 'dev_al_{}'
    MEDIAN_ITEM = 'median_al_{}'
    DMEDIAN_ITEM = 'dev_al_median_{}'
    MODE_ITEM = 'mode_al_{}'
    MAX_ITEM = 'max_al_{}'
    MIN_ITEM = 'min_al_{}'
    DMAX_ITEM = 'dev_max_al_{}'
    DMIN_ITEM = 'dev_min_al_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)

def add_devs_with3cat(df):
    """Return ."""
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']
    df['cat_age_sex'] = df[['age', 'sex', 'limit']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_three_{}'
    DEV_ITEM = 'dev_three_{}'
    MEDIAN_ITEM = 'median_three_{}'
    DMEDIAN_ITEM = 'dev_three_median_{}'
    MODE_ITEM = 'mode_three_{}'
    MAX_ITEM = 'max_three_{}'
    MIN_ITEM = 'min_three_{}'
    DMAX_ITEM = 'dev_three_max_{}'
    DMIN_ITEM = 'dev_three_min_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)

def add_devs_sex(df):
    """Return ."""
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']
    df['cat_age_sex'] = df[['limit']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_sex_{}'
    DEV_ITEM = 'dev_sex_{}'
    MEDIAN_ITEM = 'median_sex_{}'
    DMEDIAN_ITEM = 'dev_sex_median_{}'
    MODE_ITEM = 'mode_sex_{}'
    MAX_ITEM = 'max_sex_{}'
    MIN_ITEM = 'min_sex_{}'
    DMAX_ITEM = 'dev_max_sex_{}'
    DMIN_ITEM = 'dev_min_sex_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)

def add_devs_date(df):
    """Return ."""
    # arr = ['consume_count', 'consume_amount', 'loan_amount', 'loan_count', 'plannum', 'click_count']
    # got worse : plannum,loan_count
    # arr = ['consume_amount_cum', 'loan_amount_cum', 'loan_amount', 'click_count_cum', 'click_counts_sum',
    # 'click_count','consume_amounts_sum']
    df['cat_age_sex'] = df[['active_year', 'active_month']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    AVG_ITEM = 'avg_date_{}'
    DEV_ITEM = 'dev_date_{}'
    MEDIAN_ITEM = 'median_date_{}'
    DMEDIAN_ITEM = 'dev_date_median_{}'
    MODE_ITEM = 'mode_date_{}'
    MAX_ITEM = 'max_date_{}'
    MIN_ITEM = 'min_date_{}'
    DMAX_ITEM = 'dev_date_max_{}'
    DMIN_ITEM = 'dev_date_min_{}'
    for item in arr:
        this_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('mean').to_dict()
        median_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('median').to_dict()
        max_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('max').to_dict()
        min_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].aggregate('min').to_dict()
        # mode_dic = df.groupby(by=['cat_age_sex'], as_index=True)[item].apply(lambda x: x.mode()[0]).to_dict()
        avg_item = AVG_ITEM.format(item)
        dev_item = DEV_ITEM.format(item)
        d_median_item = DMEDIAN_ITEM.format(item)
        median_item = MEDIAN_ITEM.format(item)
        max_item = MAX_ITEM.format(item)
        d_max_item = DMAX_ITEM.format(item)
        min_item = MIN_ITEM.format(item)
        d_min_item = DMIN_ITEM.format(item)
        # mode_item = MODE_ITEM.format(item)
        df[avg_item] = df['cat_age_sex'].map(this_dic)
        df[median_item] = df['cat_age_sex'].map(median_dic)
        # df[max_item] = df['cat_age_sex'].map(max_dic)
        # df[min_item] = df['cat_age_sex'].map(min_dic)
        # df[mode_item] = df['cat_age_sex'].map(mode_dic)
        # df[dev_item] = (df[item] - df[avg_item])
        df[dev_item] = (df[item] - df[avg_item]) / df[avg_item]
        df[d_median_item] = 0
        df.loc[df[median_item] <> 0,d_median_item] = (df[item] - df[median_item]) / df[median_item]

        # df[d_max_item] = (df[item] - df[max_item]) / df[max_item]
        # df[d_min_item] = 0
        # df.loc[df[min_item]<> 0, d_min_item] = (df[item] - df[min_item]) / df[min_item]
        # df = df.drop([avg_item], axis=1)
    return df.drop(['cat_age_sex'], axis=1)


def add_combine_features(df):
    # 'loan_count,loan_count_cum,plannum,plannum_cum,loan_amount,loan_count'
    df['lc_per_plannum'] = 0
    df['lc_cum_per_plannum_cum'] = 0
    df['lm_per_loan'] = 0
    df['lm_cum_per_loan_cum'] = 0
    df['lm_per_plan'] = 0
    df['lm_cum_per_plan_cum'] = 0

    df['cm_per_cc'] = 0
    df['cm_cum_per_cc_cum'] = 0

    # df['comc_per_plannum'] = 0 -> useless
    # df['comc_cum_per_plannum_cum'] = 0-> useless
    # df['cc_per_loanc'] = 0-> useless
    # df['cc_cum_per_loanc_cum'] = 0-> useless
    # df['conm_per_loanm'] = 0
    # df['conm_cum_per_loanm_cum'] = 0

    # df['clc_per_comc'] = 0
    # df['clc_cum_per_comc_cum'] = 0
    df.loc[df['plannum'] <> 0, 'lc_per_plannum'] = df['loan_count']/df['plannum']
    df.loc[df['plannum_cum'] <> 0, 'lc_cum_per_plannum_cum'] = df['loan_count_cum'] / df['plannum_cum']
    df.loc[df['loan_count'] <> 0, 'lm_per_loan'] = df['loan_amount'] / df['loan_count']
    df.loc[df['loan_count_cum'] <> 0, 'lm_cum_per_loan_cum'] = df['loan_amount_cum'] / df['loan_count_cum']
    df.loc[df['plannum'] <> 0, 'lm_per_plan'] = df['loan_amount'] / df['plannum']
    df.loc[df['plannum_cum'] <> 0, 'lm_cum_per_plan_cum'] = df['loan_amount_cum'] / df['plannum_cum']

    df.loc[df['consume_count'] <> 0, 'cm_per_cc'] = df['consume_amount'] / df['consume_count']
    df.loc[df['consume_amount_cum'] <> 0, 'cm_cum_per_cc_cum'] = df['consume_amount_cum'] / df['consume_count_cum']
    # df.loc[df['plannum'] <> 0, 'comc_per_plannum'] = df['consume_count'] / df['plannum']
    # df.loc[df['plannum_cum'] <> 0, 'comc_cum_per_plannum_cum'] = df['consume_count_cum'] / df['plannum_cum']
    
    # df.loc[df['loan_count'] <> 0 ,'cc_per_loanc'] = df['consume_count'] / df['loan_count']
    # df.loc[df['loan_count_cum'] <> 0 ,'cc_cum_per_loanc_cum'] = df['consume_count_cum'] / df['loan_count_cum']
    # df.loc[df['loan_amount'] <> 0, 'conm_per_loanm'] = df['consume_amount'] / df['loan_amount']
    # df.loc[df['loan_amount_cum'] <> 0, 'conm_cum_per_loanm_cum'] = df['consume_amount_cum'] / df['loan_amount_cum']

    return df

def add_rank_features(df):
    ranks = ['consume_count','loan_count','loan_amount','consume_amount']+['consume_count_cum','loan_count_cum','loan_amount_cum','consume_amount_cum']
    RANK = '{}_rank'
    for item in ranks:
        ranks_f = RANK.format(item)
        df[ranks_f] = df[item].rank(ascending=0)
    return df

if __name__ == "__main__":
    print('begin to load data')
    train, submit = load_data()
    print('to csv ........')
    train.to_csv(INPUT_PATH + "trainv1.csv", index=False)
    submit.to_csv(INPUT_PATH + "submitv1.csv", index=False)
