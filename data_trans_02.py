import pandas as pd
import numpy as np
from pandas.tools.plotting import autocorrelation_plot, scatter_matrix
import matplotlib.pyplot as plt


file = '/home/tfl/ggprice/16-08-26 00:29:49.csv'
df = pd.read_csv(file, names=['date', 'EURUSD', 'EURJPY', 'EURAUD', 'USDJPY']).dropna()
'''
                             date           EURUSD           EURJPY  \
0      2016-08-26 00:29:49.449012   EUR/USD:1.1279  EUR/JPY:113.421
1      2016-08-26 00:29:49.757651   EUR/USD:1.1279  EUR/JPY:113.421
2        2016-08-26 00:29:50.9805   EUR/USD:1.1279  EUR/JPY:113.421
                EURAUD           USDJPY
0      EUR/AUD:1.48175  USD/JPY:100.559
1      EUR/AUD:1.48175  USD/JPY:100.559
2      EUR/AUD:1.48175  USD/JPY:100.559
'''

del df['EURJPY'], df['EURAUD'], df['USDJPY']
'''
                             date           EURUSD
0      2016-08-26 00:29:49.449012   EUR/USD:1.1279
1      2016-08-26 00:29:49.757651   EUR/USD:1.1279
2        2016-08-26 00:29:50.9805   EUR/USD:1.1279
'''

df['EURUSD'] = df['EURUSD'].apply(lambda x: float(x.split(':')[1]))
'''
                             date   EURUSD
0      2016-08-26 00:29:49.449012  1.12790
1      2016-08-26 00:29:49.757651  1.12790
2        2016-08-26 00:29:50.9805  1.12790
'''

df['date'] = df['date'].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][0:3])
'''
                          date   EURUSD
0      2016-08-26 00:29:49.449  1.12790
1      2016-08-26 00:29:49.757  1.12790
2      2016-08-26 00:29:50.980  1.12790
'''

df.index = pd.DatetimeIndex(df['date'])  # index as DatetimeIndex
'''
                                            date   EURUSD
2016-08-26 00:29:49.449  2016-08-26 00:29:49.449  1.12790
2016-08-26 00:29:49.757  2016-08-26 00:29:49.757  1.12790
2016-08-26 00:29:50.980  2016-08-26 00:29:50.980  1.12790
'''

del df['date']
'''
                          EURUSD
2016-08-26 00:29:49.449  1.12790
2016-08-26 00:29:49.757  1.12790
2016-08-26 00:29:50.980  1.12790
'''

df = df['EURUSD'].resample('1S').mean()  # resample 1S
'''
2016-08-26 00:29:49    1.127900
2016-08-26 00:29:50    1.127900
2016-08-26 00:29:51    1.127900
'''

df = pd.DataFrame(df, columns=['EURUSD'])
'''
                       EURUSD
2016-08-26 00:29:49  1.127900
2016-08-26 00:29:50  1.127900
2016-08-26 00:29:51  1.127900
'''

df['later_30S'] = df['EURUSD'].shift(-30)
df['d_30S'] = df['later_30S'] - df['EURUSD']

df['bd_30S_up'] = 0
df['bd_30S_hold'] = 0
df['bd_30S_down'] = 0
df.loc[df.d_30S > 0, 'bd_30S_up'] = 1
df.loc[df.d_30S == 0, 'bd_30S_hold'] = 1
df.loc[df.d_30S < 0, 'bd_30S_down'] = 1

df['logr_30S'] = np.log(df['later_30S'] / df['EURUSD'])

df['ma_15S'] = df['EURUSD'].rolling(15).mean()  # 15 seconds moving average

df['ma_30S'] = df['EURUSD'].rolling(30).mean()  # 30 seconds moving average

df['ma_60S'] = df['EURUSD'].rolling(60).mean()  # 60 seconds moving average

df['ma_180S'] = df['EURUSD'].rolling(180).mean()  # 180 seconds moving average

df['ma_300S'] = df['EURUSD'].rolling(300).mean()  # 300 seconds moving average

df['ma_600S'] = df['EURUSD'].rolling(600).mean()  # 600 seconds moving average

#  11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
# df['later_60S'] = df['EURUSD'].shift(-60)
# df['later_120S'] = df['EURUSD'].shift(-120)
# df['later_300S'] = df['EURUSD'].shift(-300)

df['gold'] = 0

df['expire30'] = 0

index30 = pd.date_range(start='2016-08-26', end='2016-08-27', freq='30S')
df.loc[df.index & index30, 'expire30'] = 1

# df[['EURUSD', 'ma_15S', 'ma_30S', 'ma_60S', 'ma_180S', 'ma_300S', 'ma_600S']].plot()
df.loc[
    (df.ma_15S > df.ma_30S) &
    (df.ma_30S > df.ma_60S) &
    (df.ma_60S > df.ma_180S) &
    (df.ma_180S > df.ma_300S) &
    (df.ma_300S > df.ma_600S), 'gold'
] = 1

signal = df.ix[(df.gold == 1) & (df.expire30 == 1)]
sele = df.ix[(df.gold == 1) & (df.bd_30S_down == 1) & (df.expire30 == 1)]

