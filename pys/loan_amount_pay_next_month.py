# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'


def load_data():

    users = pd.read_csv(INPUT_PATH + 't_user.csv')
    loans = pd.read_csv(INPUT_PATH + 't_loan.csv')

    # origin price
    loans.loan_amount = 5**loans.loan_amount - 1
    users.limit = 5 ** users.limit -1

    # loans
    loans['Date'] = pd.to_datetime(loans['loan_time'], errors='coerce')
    loans['transaction_month'] = loans['Date'].dt.month

    loans['lam_plan'] = loans['loan_amount'] / loans['plannum']

    loans['8_lmp_pay'] = 0
    loans['9_lmp_pay'] = 0
    loans['10_lmp_pay'] = 0
    loans['11_lmp_pay'] = 0

    loans['8_lmp_reste'] = 0
    loans['9_lmp_reste'] = 0
    loans['10_lmp_reste'] = 0
    loans['11_lmp_reste'] = 0

    loans.loc[loans.transaction_month == 8, '8_lmp_reste'] = loans.loan_amount

    loans.loc[loans.transaction_month == 8, '9_lmp_pay'] = loans.lam_plan
    loans.loc[(loans.transaction_month == 8) & (loans.plannum>1), '9_lmp_reste'] = (loans.plannum-1) * loans.lam_plan
    loans.loc[loans.transaction_month == 9, '9_lmp_reste'] = loans.loan_amount

    loans.loc[(loans.transaction_month == 8) & (loans.plannum>1), '10_lmp_pay'] = loans.lam_plan
    loans.loc[loans.transaction_month == 9, '10_lmp_pay'] = loans.lam_plan
    loans.loc[(loans.transaction_month == 8) & (loans.plannum>2), '10_lmp_reste'] =  (loans.plannum-2) * loans.lam_plan
    loans.loc[(loans.transaction_month == 9) & (loans.plannum>1), '10_lmp_reste'] =  (loans.plannum-1) * loans.lam_plan
    loans.loc[loans.transaction_month == 10, '10_lmp_reste'] = loans.loan_amount

    loans.loc[(loans.transaction_month == 8) & (loans.plannum>2), '11_lmp_pay'] = loans.lam_plan
    loans.loc[(loans.transaction_month == 9) & (loans.plannum>1), '11_lmp_pay'] = loans.lam_plan
    loans.loc[loans.transaction_month == 10, '11_lmp_pay'] = loans.lam_plan
    loans.loc[(loans.transaction_month == 8) & (loans.plannum>3), '11_lmp_reste'] =  (loans.plannum-3) * loans.lam_plan
    loans.loc[(loans.transaction_month == 9) & (loans.plannum>2), '11_lmp_reste'] =  (loans.plannum-2) * loans.lam_plan
    loans.loc[(loans.transaction_month == 10) & (loans.plannum>1), '11_lmp_reste'] =  (loans.plannum-1) * loans.lam_plan
    loans.loc[loans.transaction_month == 11, '11_lmp_reste'] = loans.loan_amount

    loans_new = loans.groupby(by=["uid"], as_index=False)["8_lmp_pay",'8_lmp_reste', "9_lmp_pay", '9_lmp_reste',
    '10_lmp_pay','10_lmp_reste','11_lmp_pay','11_lmp_reste'].sum()

    return loans_new

if __name__ == "__main__":
    print('begin to load data')
    loan_pay_next= load_data()
    print(' load data finished')
    print('to csv ........')
    loan_pay_next.to_csv(OUTPUT_PATH +"loan_pay_next.csv", index=False)
