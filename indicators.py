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

#bollinger bands indicator 

#ichimoku cloud indicator 

#simple moving average 
def simple_moving_average(data,ndays):
    SMA=pd.Series(pd.rolling_mean(data['Close'],n),name='SMA')
    data=data.join(SMA)
    return data
#money flow index

#envelops