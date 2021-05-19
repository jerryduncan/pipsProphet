import os
import numpy as np 
import pandas as pd 
import glob



if __name__ == '__main__':
    cur = ['BTCUSD', 'DOGEUSD', 'EURCAD', 'EURJPY', 'EURUSD', 'GBPUSD', 'USDJPY']
    focus_cur = ['EURCAD', 'EURJPY', 'EURUSD', 'GBPUSD']
    date = []
    clean_nan(cur)