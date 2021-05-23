from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

from functools import wraps
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame
from pandas.stats import moments 

#stochastic oscillator indator 
def stochastic_oscillator_a(df):
    SOa = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name='SO%a')
    df = df.join(SOa)
    return df 

def stochastic_oscillator_b(df, n):
    """Calculate stochastic oscillator %B for given data
    """
    SOa = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name='SO%a')
    SOb = pd.Series(SOa.ewm(span=n, min_periods=n).mean(), name='SO%b')
    df = df.join(SOb)
    return df

#bollinger bands indicator 
def bollinger_bands(df, n, std, add_average=True):
    average = df['close'].rolling(window=n, center=False).mean()
    sd = df['close'].rolling(window=n, center=False).std()
    up_bollinger = pd.Series(average + (sd * std), name='bband_upper_' + str(n))
    dn_bollinger = pd.Series(average - (sd * std), name='bband_lower_' + str(n))

    if add_average:
        average = pd.Series(average, name='bband_average_' + str(n))
        df = df.join(pd.concat([up_bollinger, dn_bollinger, average], axis=1))
    else:
        df = df.join(pd.concat([up_bollinger, dn_bollinger], axis=1))

    return df


#ichimoku cloud indicator 
def ichimoku(s, n1=9, n2=26, n3=52):
    conv = (hhv(s, n1) + llv(s, n1)) / 2
    base = (hhv(s, n2) + llv(s, n2)) / 2

    span_a = (conv + base) / 2
    span_b = (hhv(s, n3) + llv(s, n3)) / 2

    return DataFrame(dict(conv=conv, base=base, span_a=span_a.shift(n2),
                          span_b=span_b.shift(n2), lspan=s.close.shift(-2)))


#simple moving average 
def simple_moving_average(data,ndays):
    SMA=pd.Series(pd.rolling_mean(data['Close'],n),name='SMA')
    data=data.join(SMA)
    return data
#money flow index

#envelops