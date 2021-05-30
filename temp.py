import pandas as pd
import talib

df= pd.read_csv('')

close = df['close'].astype('float')
volume = df['volume'].astype('float')
obv = talib.MA(close, volume)
print(obv)