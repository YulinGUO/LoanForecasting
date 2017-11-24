# -*- coding: utf-8 -*-
import pandas as pd
import datasplit as ds
from sklearn.decomposition import TruncatedSVD

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

def load_data():
	users = pd.read_csv(INPUT_PATH + 't_user.csv')
	clicks = pd.read_csv(INPUT_PATH + 't_click.csv')
	clicks['Date'] = pd.to_datetime(clicks['click_time'], errors='coerce')
	clicks['transaction_month'] = clicks['Date'].dt.month
	clicks["p_count"] = 1
	clicks = clicks.groupby(by=["uid", "transaction_month", "pid", "param"], as_index=False)["p_count"].sum()
	clicks_new = clicks.pivot_table(['p_count'], ['uid'], ['transaction_month', "pid", "param"], fill_value=0)
	clicks_new.reset_index(drop=False, inplace=True)
	clicks_new.columns = ['{}_pid_{}_param_{}'.format(i[1], i[2], i[3]) for i in clicks_new.columns]
	clicks_new = clicks_new.rename(index=str, columns={"_pid__param_": "uid"})

	param_list = clicks.param.map(lambda x:"param_"+str(x)).unique()
	pid_list = clicks.pid.map(lambda x:"pid_"+str(x)).unique()
	pid_param_list = []
	for pid in pid_list:
	    for param in param_list:
	        pid_param_list.append(pid+"_"+param)

	df8 = get_df_by_month(clicks_new, '8')
	df8 = df8.rename(columns=ds.remove_month_rename)
	df8 = add_default_param(df8, pid_param_list)
	df8 = users_merge(users, df8, 8)

	df9 = get_df_by_month(clicks_new, '9')
	df9 = df9.rename(columns=ds.remove_month_rename)
	df9 = add_default_param(df9, pid_param_list)
	df9 = users_merge(users, df9, 9)

	df10 = get_df_by_month(clicks_new, '10')
	df10 = df10.rename(columns=ds.remove_month_rename)
	df10 = add_default_param(df10, pid_param_list)
	df10 = users_merge(users, df10, 10)

	df11 = get_df_by_month(clicks_new, '11')
	df11 = df11.rename(columns=ds.remove_month_rename)
	df11 = add_default_param(df11, pid_param_list)
	df11 = users_merge(users, df11, 11)

	frames = [df8, df9, df10, df11]
	data = pd.concat(frames)
	data.reset_index(drop=True, inplace=True)

	for col in data:
	    data[col] = data[col].fillna(0)

	return data
	
def get_pid_cols(df):
    return df.columns[df.columns.str.startswith("pid")]

def get_df_by_month(df, month):
    """Return ."""
    cols = ds.get_column_by_month(df, month)
    basic = ['uid']

    all_cols = basic + cols 
    return df[all_cols]

def add_default_param(df, param_list):
    for col in param_list:
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
	print(' load data finished')
	col_num = 3
	svd = TruncatedSVD(col_num)
	name_basic = 'pid_param_{}'
	cols_svd_name = map(lambda x: name_basic.format(x), range(0, col_num))
	new_data = svd.fit_transform(data[get_pid_cols(data)])
	new_df = pd.DataFrame(new_data, columns=cols_svd_name)

	res = data.join(new_df)
	cols = ["uid", "transaction_month"] + cols_svd_name
	res = res[cols]
	train, submit = split_param(res)
	print('to csv ........')
	train.to_csv(INPUT_PATH +"train_param.csv", index=False)
	submit.to_csv(INPUT_PATH +"submit_param.csv", index=False)
