#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

users = pd.read_csv(INPUT_PATH + 't_user.csv')
orders = pd.read_csv(INPUT_PATH + 't_order.csv')
loans = pd.read_csv(INPUT_PATH + 't_loan.csv')
loans_sum = pd.read_csv(INPUT_PATH + 't_loan_sum.csv')
clicks = pd.read_csv(INPUT_PATH + 't_click.csv')


