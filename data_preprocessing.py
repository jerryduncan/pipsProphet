import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 

class Extractor:
    def __init__(self, df):
        self.df = df
        self.open = df['open'].astype('float')
        self.high = df['high'].astype('float')
        self.low = df['low'].astype('float')
        self.close = df['close'].astype('float')
        self.volume = df['volume'].astype('float')

    def add_bar_features(self):
        self.df['bar_ho'] = self.high - self.open
        self.df['bar_hl'] = self.high - self.low
        self.df['bar_hc'] = self.high - self.close
        self.df['bar_co'] = self.close - self.open
        self.df['bar_ol'] = self.open - self.low
        self.df['bar_cl'] = self.close - self.low
        self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1)

        return self.df  