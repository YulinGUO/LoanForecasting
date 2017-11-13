import pandas as pd
import numpy as np

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

users = pd.read_csv(INPUT_PATH + 't_user.csv')
orders = pd.read_csv(INPUT_PATH + 't_order.csv')
loans = pd.read_csv(INPUT_PATH + 't_loan.csv')
loans_sum = pd.read_csv(INPUT_PATH + 't_loan_sum.csv')
clicks = pd.read_csv(INPUT_PATH + 't_click.csv')

loans.loan_amount = 5**loans.loan_amount
loans_sum.loan_sum  = 5 ** loans_sum.loan_sum



loans['Date'] = pd.to_datetime(loans['loan_time'], errors='coerce')
loans['transaction_month'] = loans['Date'].dt.month
loan_of_nov = loans.loc[loans.transaction_month==11].groupby(by='uid', as_index=False).sum()

loan_of_nov.merge(loans_sum, how='left', on='uid')

#check loan_amount and loan_sum