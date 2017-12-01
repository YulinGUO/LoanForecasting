# -*- coding: utf-8 -*-
import pandas as pd
import datasplit as ds
from sklearn.decomposition import TruncatedSVD

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

def load_data():
    users = pd.read_csv(INPUT_PATH + 't_user.csv')
    orders = pd.read_csv(INPUT_PATH + 't_order.csv')
    orders['Date'] = pd.to_datetime(orders['buy_time'], errors='coerce')
    orders['transaction_month'] = orders['Date'].dt.month
    #orders["click_count"] = 1
    orders = orders.groupby(by=["uid", "transaction_month", "cate_id"], as_index=False)["qty"].sum()
    orders_new = orders.pivot_table(['qty'], ['uid'], ['transaction_month', "cate_id"], fill_value=0)
    orders_new.reset_index(drop=False, inplace=True)
    orders_new.columns = ['{}_cate_id_{}'.format(i[1], i[2]) for i in orders_new.columns]
    orders_new = orders_new.rename(index=str, columns={"_cate_id_": "uid"})

    cate_list = orders.cate_id.map(lambda x:"cate_id_"+str(x)).unique()

    df8 = get_df_by_month(orders_new, '8')
    df8 = df8.rename(columns=ds.remove_month_rename)
    df8 = add_default_cate(df8, cate_list)
    df8 = users_merge(users, df8, 8)
    df8 = add_cols_count(df8,cate_list)

    df9 = get_df_by_month(orders_new, '9')
    df9 = df9.rename(columns=ds.remove_month_rename)
    df9 = add_default_cate(df9, cate_list)
    df9 = users_merge(users, df9, 9)
    df9 = add_cols_count(df9,cate_list)

    df10 = get_df_by_month(orders_new, '10')
    df10 = df10.rename(columns=ds.remove_month_rename)
    df10 = add_default_cate(df10, cate_list)
    df10 = users_merge(users, df10, 10)
    df10 = add_cols_count(df10,cate_list)

    df11 = get_df_by_month(orders_new, '11')
    df11 = df11.rename(columns=ds.remove_month_rename)
    df11 = add_default_cate(df11, cate_list)
    df11 = users_merge(users, df11, 11)
    df11 = add_cols_count(df11,cate_list)

    frames = [df8, df9, df10, df11]
    data = pd.concat(frames)
    data.reset_index(drop=True, inplace=True)

    for col in data:
        data[col] = data[col].fillna(0)

    return data
    
def get_cate_cols(df):
    return df.columns[df.columns.str.startswith("cate_id")]

def get_df_by_month(df, month):
    """Return ."""
    cols = ds.get_column_by_month(df, month)
    basic = ['uid']

    all_cols = basic + cols 
    return df[all_cols]

def add_default_cate(df, cate_list):
    for col in cate_list:
        if col not in df:
            df[col] = 0
    return df

def users_merge(users, df, month):
    new_df = users[["uid"]].merge(df, on="uid", how="left")
    new_df["transaction_month"] = month
    return new_df

def split_cate(df):
    cate8 = df.iloc[0:len(res)*1/4]
    cate9 = df.iloc[len(res)*1/4:len(res)*1/2]
    cate10 = df.iloc[len(res)*1/2:len(res)*3/4]
    cate11 = df.iloc[len(res)*3/4:len(res)]
    cate_frames = [cate8, cate9, cate10]
    train = pd.concat(cate_frames)
    return train, cate11

def add_cols_count(df, cate_list):
    df['cat_counts'] = df[cate_list].gt(0).sum(axis=1)
    return df

if __name__ == "__main__":
    print('begin to load data')
    data = load_data()
    print(' load data finished')
    col_num = 3
    svd = TruncatedSVD(col_num)
    name_basic = 'cate_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))
    new_data = svd.fit_transform(data[get_cate_cols(data) ])
    new_df = pd.DataFrame(new_data, columns=cols_svd_name)

    res = data.join(new_df)
    cols = ["uid", "transaction_month", 'cat_counts'] + cols_svd_name
    res = res[cols]
    train, submit = split_cate(res)
    print('to csv ........')
    train.to_csv(INPUT_PATH +"train_cate_id.csv", index=False)
    submit.to_csv(INPUT_PATH +"submit_cate_id.csv", index=False)
