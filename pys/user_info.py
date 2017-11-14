import pandas as pd
INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

loans_sum = pd.read_csv(INPUT_PATH + "t_loan_sum.csv")
clicks = pd.read_csv(INPUT_PATH + "t_click.csv")
loans = pd.read_csv(INPUT_PATH +"t_loan.csv")
orders = pd.read_csv(INPUT_PATH +"t_order.csv")
users = pd.read_csv(INPUT_PATH +"t_user.csv")

loans["loan_amount_new"] = loans["loan_amount"].map(lambda x:5**x)
orders["price_new"] = orders["price"].map(lambda x:5**x)
orders["discount_new"] = orders["discount"].map(lambda x:5**x-1)
users["limit"] = users["limit"].map(lambda x:5**x)

loans['Date'] = pd.to_datetime(loans['loan_time'], errors='coerce')
loans['transaction_month'] = loans['Date'].dt.month
orders['Date'] = pd.to_datetime(orders['buy_time'], errors='coerce')
orders['transaction_month'] = orders['Date'].dt.month
clicks['Date'] = pd.to_datetime(clicks['click_time'], errors='coerce')
clicks['transaction_month'] = clicks['Date'].dt.month

loans["loan_8"] = loans["loan_amount_new"][loans["transaction_month"] == 8]
loans["loan_9"] = loans["loan_amount_new"][loans["transaction_month"] == 9]
loans["loan_10"] = loans["loan_amount_new"][loans["transaction_month"] == 10]
loans["loan_11"] = loans["loan_amount_new"][loans["transaction_month"] == 11]
loans["loan_8"] = loans["loan_8"].fillna(0)
loans["loan_9"] = loans["loan_9"].fillna(0)
loans["loan_10"] = loans["loan_10"].fillna(0)
loans["loan_11"] = loans["loan_11"].fillna(0)
loans["count"] = 1
loans["loan_count_8"] = loans["count"][loans["transaction_month"] == 8]
loans["loan_count_9"] = loans["count"][loans["transaction_month"] == 9]
loans["loan_count_10"] = loans["count"][loans["transaction_month"] == 10]
loans["loan_count_11"] = loans["count"][loans["transaction_month"] == 11]
loans["loan_count_8"] = loans["loan_count_8"].fillna(0)
loans["loan_count_9"] = loans["loan_count_9"].fillna(0)
loans["loan_count_10"] = loans["loan_count_10"].fillna(0)
loans["loan_count_11"] = loans["loan_count_11"].fillna(0)
loans_month_count = loans.groupby(by=["uid"], as_index=False)["loan_count_8", "loan_count_9", "loan_count_10", "loan_count_11"].sum()
loans_month = loans.groupby(by=["uid"], as_index=False)["loan_8", "loan_9", "loan_10", "loan_11"].sum()

orders["consume_amount"] = orders["price_new"]*orders["qty"]-orders["discount_new"]
orders["consume_8"] = orders["consume_amount"][orders["transaction_month"] == 8]
orders["consume_9"] = orders["consume_amount"][orders["transaction_month"] == 9]
orders["consume_10"] = orders["consume_amount"][orders["transaction_month"] == 10]
orders["consume_11"] = orders["consume_amount"][orders["transaction_month"] == 11]
orders["consume_8"] = orders["consume_8"].fillna(0)
orders["consume_9"] = orders["consume_9"].fillna(0)
orders["consume_10"] = orders["consume_10"].fillna(0)
orders["consume_11"] = orders["consume_11"].fillna(0)
orders["count"] = 1
orders["consume_count_8"] = orders["count"][orders["transaction_month"] == 8]
orders["consume_count_9"] = orders["count"][orders["transaction_month"] == 9]
orders["consume_count_10"] = orders["count"][orders["transaction_month"] == 10]
orders["consume_count_11"] = orders["count"][orders["transaction_month"] == 11]
orders["consume_count_8"] = orders["consume_count_8"].fillna(0)
orders["consume_count_9"] = orders["consume_count_9"].fillna(0)
orders["consume_count_10"] = orders["consume_count_10"].fillna(0)
orders["consume_count_11"] = orders["consume_count_11"].fillna(0)
consumes_month_count = orders.groupby(by=["uid"], as_index=False)["consume_count_8", "consume_count_9", "consume_count_10", "consume_count_11"].sum()
consumes_month = orders.groupby(by=["uid"], as_index=False)["consume_8", "consume_9", "consume_10", "consume_11"].sum()

clicks["count"] = 1
clicks["click_count_8"] = clicks["count"][clicks["transaction_month"]==8]
clicks["click_count_9"] = clicks["count"][clicks["transaction_month"]==9]
clicks["click_count_10"] = clicks["count"][clicks["transaction_month"]==10]
clicks["click_count_11"] = clicks["count"][clicks["transaction_month"]==11]
clicks["click_count_8"] = clicks["click_count_8"].fillna(0)
clicks["click_count_9"] = clicks["click_count_9"].fillna(0)
clicks["click_count_10"] = clicks["click_count_10"].fillna(0)
clicks["click_count_11"] = clicks["click_count_11"].fillna(0)
clicks_month_count = clicks.groupby(by=["uid"], as_index=False)["click_count_8", "click_count_9", "click_count_10", "click_count_11"].sum()

loans_info = pd.merge(loans_month, loans_month_count, on=["uid"])
consume_info = pd.merge(consumes_month, consumes_month_count, on=["uid"])
user_info = pd.merge(users, pd.merge(consume_info, pd.merge(clicks_month_count, loans_info, on=["uid"], how="left"), on=["uid"], how="left"), on=["uid"], how="left")

user_info.to_csv(OUTPUT_PATH +"user_info.csv", index=False)