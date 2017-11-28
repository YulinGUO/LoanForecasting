# -*- coding: utf-8 -*-
import pandas as pd
import datasplit as ds
from sklearn.decomposition import TruncatedSVD

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

def load_data():
    users = pd.read_csv(INPUT_PATH + 't_user.csv')
    loans = pd.read_csv(INPUT_PATH + 't_loan.csv')
    loans.loan_amount = 5**loans.loan_amount - 1
    loans['Date'] = pd.to_datetime(loans['loan_time'], errors='coerce')
    loans['transaction_month'] = loans['Date'].dt.month
    loans["day_of_week"] = loans.Date.dt.dayofweek
    loans.loan_amount = loans.loan_amount.fillna(0)
    loans["loan_count"] = 1
    loans_new = loans.groupby(by=["uid", "transaction_month", "day_of_week"], as_index=False)["loan_amount", "loan_count"].sum()
    loans_new = loans_new.pivot_table(['loan_amount', "loan_count"], ['uid'], ['transaction_month', "day_of_week"], fill_value=0)
    loans_new.reset_index(drop=False, inplace=True)
    loans_new.columns = ['{}_dow_{}_{}'.format(i[1], i[2], i[0]) for i in loans_new.columns]
    loans_new = loans_new.rename(index=str, columns={"_dow__uid": "uid"})

    col_list = []
    for col in ["loan_amount", "loan_count"]:
        for dow in range(7):
            col_list.append("dow_{}_{}".format(dow, col))

    df8 = get_df_by_month(loans_new, '8')
    df8 = df8.rename(columns=ds.remove_month_rename)
    df8 = add_default_col(df8, col_list)
    df8 = users_merge(users, df8, 8)
    ###weekends mean sunday&monday
    df8["workdays"] = 22
    df8["weekends"] = 8
    
    df9 = get_df_by_month(loans_new, '9')
    df9 = df9.rename(columns=ds.remove_month_rename)
    df9 = add_default_col(df9, col_list)
    df9 = users_merge(users, df9, 9)
    df9["workdays"] = 21
    df9["weekends"] = 10
   
    df10 = get_df_by_month(loans_new, '10')
    df10 = df10.rename(columns=ds.remove_month_rename)
    df10 = add_default_col(df10, col_list)
    df10 = users_merge(users, df10, 10)
    df10["workdays"] = 22
    df10["weekends"] = 8
    
    df11 = get_df_by_month(loans_new, '11')
    df11 = df11.rename(columns=ds.remove_month_rename)
    df11 = add_default_col(df11, col_list)
    df11 = users_merge(users, df11, 11)
    df11["workdays"] = 23
    df11["weekends"] = 8
    
    frames = [df8, df9, df10, df11]
    data = pd.concat(frames)
    data.reset_index(drop=True, inplace=True)

    for col in data:
        data[col] = data[col].fillna(0)
        
    return data

def get_dow_cols(df):
    return df.columns[df.columns.str.startswith("dow")]

def get_df_by_month(df, month):
    """Return ."""
    cols = ds.get_column_by_month(df, month)
    basic = ['uid']

    all_cols = basic + cols 
    return df[all_cols]

def add_default_col(df, col_list):
    for col in col_list:
        if col not in df:
            df[col] = 0
    return df

def users_merge(users, df, month):
    new_df = users[["uid"]].merge(df, on="uid", how="left")
    new_df["transaction_month"] = month
    return new_df

def split_param(df):
    param8 = df.iloc[0:len(res)*1/4]
    param9 = df.iloc[len(res)*1/4:len(res)*1/2]
    param10 = df.iloc[len(res)*1/2:len(res)*3/4]
    param11 = df.iloc[len(res)*3/4:len(res)]
    param_frames = [param8, param9, param10]
    #param_frames = [param9, param10]
    train = pd.concat(param_frames)
    return train, param11

if __name__ == "__main__":
    print('begin to load data')
    data = load_data()
    col_num = 3
    svd = TruncatedSVD(col_num)
    name_basic = 'dow_{}'
    cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))
    new_data = svd.fit_transform(data[get_dow_cols(data)])
    new_df = pd.DataFrame(new_data, columns=cols_svd_name)

    res = data.join(new_df)
    cols = ["uid", "transaction_month", "workdays", "weekends"] + cols_svd_name
    res = res[cols]
    train, submit = split_param(res)
    print('to csv ........')
    train.to_csv(INPUT_PATH +"train_day_of_week.csv", index=False)
    submit.to_csv(INPUT_PATH +"submit_day_of_week.csv", index=False)
