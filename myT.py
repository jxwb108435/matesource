import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
# import re
from itertools import groupby
import os


def newestfile():
    prepath = '/home/lin/ggprice/'
    file_ar = os.listdir(prepath)  # list all files

    df_file = pd.DataFrame(file_ar, columns=['filename'])
    df_file['newfile'] = df_file['filename'].apply(lambda x: x.split('.')[0])
    df_file['date'] = df_file['newfile']. \
        apply(lambda x: int(x.split(' ')[0].replace('-', '') + x.split(' ')[1].replace(':', '')))

    df_file = df_file.sort_values(by='date')
    df_file.index = range(len(df_file))

    _file_target = prepath + df_file['filename'][-1:].values[0]  # find newest file

    return _file_target


def live_data(_file_target):
    normed = lambda x: (x - x.mean()) / x.std()

    df = pd.read_csv(_file_target, names=['date', 'EURUSD', 'EURJPY', 'EURAUD', 'USDJPY']).dropna()
    df = df[-3500:]

    df['EURUSD'] = df['EURUSD'].apply(lambda x: float(x.split(':')[1]))
    df['date'] = df['date'].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][0:3])

    df.index = pd.DatetimeIndex(df['date'])  # index as DatetimeIndex
    df = df.resample('1S').mean()  # resample 1S

    df['sma20'] = df['EURUSD'].rolling(20).mean()
    df['sma60'] = df['EURUSD'].rolling(60).mean()
    df['sma180'] = df['EURUSD'].rolling(180).mean()

    df = df.dropna()

    df = df.ix[-601:]  # keep 10 minutes price

    _tmp1 = np.hstack([normed(df['sma20'].values), normed(df['sma60'].values), normed(df['sma180'].values)])

    return _tmp1


def gen_xy(_file_ar):
    _input_tick, _label_tick = 601, 121
    _input_ar, _label_ar = [], []
    for each_file in _file_ar:
        df = pd.read_csv(each_file, names=['date', 'EURUSD', 'EURJPY', 'EURAUD', 'USDJPY']).dropna()
        del df['EURJPY'], df['EURAUD'], df['USDJPY']

        df['EURUSD'] = df['EURUSD'].apply(lambda x: float(x.split(':')[1]))
        df['date'] = df['date'].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][0:3])

        df.index = pd.DatetimeIndex(df['date'])  # index as DatetimeIndex
        df = df.resample('1S').mean()  # resample 1S

        df['sma20'] = df['EURUSD'].rolling(20).mean()
        df['sma60'] = df['EURUSD'].rolling(60).mean()
        df['sma180'] = df['EURUSD'].rolling(180).mean()

        df = df.dropna()

        for _sp in range(0, len(df) - _input_tick):
            _tmp1 = np.hstack([df['sma20'][_sp:_sp + _input_tick].values,
                               df['sma60'][_sp:_sp + _input_tick].values,
                               df['sma180'][_sp:_sp + _input_tick].values])
            _tmp2 = df['sma20'][_sp + _input_tick + 1:_sp + _input_tick + _label_tick + 1].values

            if (1803 == len(_tmp1)) and (121 == len(_tmp2)):
                _input_ar.append(_tmp1)
                _label_ar.append(_tmp2)

    return _input_ar, _label_ar


def gen_file(_num_test=2):
    prepath = '/home/tfl/ggprice/'
    file_ar = os.listdir(prepath)  # list all files

    df_file = pd.DataFrame(file_ar, columns=['filename'])
    df_file['newfile'] = df_file['filename'].apply(lambda x: x.split('.')[0])
    df_file['date'] = df_file['newfile']. \
        apply(lambda x: int(x.split(' ')[0].replace('-', '') + x.split(' ')[1].replace(':', '')))
    df_file = df_file.sort_values(by='date')
    df_file.index = range(len(df_file))
    df_file = df_file[:-1]
    df_file['filename'] = df_file['filename'].apply(lambda x: prepath + x)

    _train_file = df_file['filename'][:-_num_test].values
    _test_file = df_file['filename'][-_num_test:].values
    return _train_file, _test_file


def get_price(asset='EURUSD', day='20090501'):
    prepath = '/home/tfl/'
    file_ar = os.listdir(prepath + asset + '/' + day[2:4] + '/')
    day_ar = list(map(lambda x: x[0:8], file_ar))

    if day in day_ar:
        df = pd.read_csv(prepath + asset + '/' + day[2:4] + '/' + day + '.csv', names=['date', 'price'])
        df = df.dropna()
        df.index = pd.date_range(df['date'][0], periods=len(df), freq='500L')
        del df['date']

        return df

    else:
        print(day, 'is not exist! check the date!')
        return False


def getdf(_filename, _asset='EURUSD'):
    df = pd.read_csv(_filename, names=['date', 'EURUSD', 'EURJPY', 'EURAUD', 'USDJPY']).dropna()
    df[_asset] = df[_asset].apply(lambda x: float(x.split(':')[1]))
    df['date'] = df['date'].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][0:3])

    df.index = pd.DatetimeIndex(df['date'])  # index as DatetimeIndex
    df = df.resample('1S').mean()  # resample 1S
    df['Price'] = df[_asset]
    del df[_asset]

    dmax = df['Price'].values.max()
    dmin = df['Price'].values.min()
    df['0_100'] = ((df['Price'] - dmin) / (dmax - dmin) * 100) + 1e-6

    return df['Price'], df['0_100'], dmax, dmin


def ln_return(_ar):
    _df = pd.DataFrame(_ar, columns=['price'])
    _df['price+1'] = _df['price'].shift(+1)
    _df['lnreturn'] = np.log(_df['price']) - np.log(_df['price+1'])

    return _df['lnreturn']


def simple_return(_ar):
    _pds = pd.DataFrame(_ar, columns=['price'])
    _pds['price -1'] = _pds['price'] - _pds['price'].shift(1)
    _pds['sreturn'] = _pds['price -1'] / _pds['price']

    return _pds['sreturn'].values


def t_1900_b2(_br):
    """ Louis Bachelier 1900
    :param _br:
    :return:
    """
    _df = pd.DataFrame([[_k, len(list(_g))] for _k, _g in groupby(_br)], columns=['key', 'num'])

    _up = _df.ix[_df.key == 1]['num'].values.sum()
    _down = _df.ix[_df.key == 0]['num'].values.sum()
    _u = len(_df)

    _numerator = 2 * _up * _down * (2 * _up * _down - _up - _down)
    _denominator = ((_up + _down) ** 2) * (_up + _down - 1)

    _x = 2 * _up * _down / (_up + _down) + 1
    _s = (_numerator / _denominator) ** 0.5
    _z = (abs(_u - _x) - 0.5) / _s

    return _z


def t_1900_b3(_br):
    """ Louis Bachelier 1900
    :param _br:
    :return:
    """
    _df = pd.DataFrame([[_k, len(list(_g))] for _k, _g in groupby(_br)], columns=['key', 'num'])

    _up = _df.ix[_df.key == 1]['num'].values.sum()
    _down = _df.ix[_df.key == -1]['num'].values.sum()
    _u = len(_df.ix[_df.key != 0])

    _numerator = 2 * _up * _down * (2 * _up * _down - _up - _down)
    _denominator = ((_up + _down) ** 2) * (_up + _down - 1)

    _x = 2 * _up * _down / (_up + _down) + 1
    _s = (_numerator / _denominator) ** 0.5
    _z = (abs(_u - _x) - 0.5) / _s

    return _z


def to_binary3(_ar):
    _df = pd.DataFrame(_ar, columns=['price'])
    _df['dif'] = _df['price'] - _df['price'].shift(1)

    _df['order'] = 0
    _df.loc[_df.dif > 0, 'order'] = 1  # go up
    _df.loc[_df['dif'] < 0, 'order'] = -1  # go down

    return _df['order'].values


def call_option_pricer(_spot, _strike, _maturity, _r, _vol):
    """ Black - Scholes
    Black - Scholes
    :param _spot:
    :param _strike:
    :param _maturity:
    :param _r:
    :param _vol:
    :return:
    """
    _d1 = (log(_spot / _strike) + (_r + 0.5 * _vol * _vol) * _maturity) / _vol / sqrt(_maturity)
    _d2 = _d1 - _vol * sqrt(_maturity)
    price = _spot * norm.cdf(_d1) - _strike * exp(-_r * _maturity) * norm.cdf(_d2)

    return price


def mbfxt_ohlc(_ohlc, _len=7, _filter=0.0):
    _bars = len(_ohlc)
    line_all = np.zeros(_bars)  # line
    line_down = np.zeros(_bars)  # down
    line_up = np.zeros(_bars)  # up

    ld_8, ld_112, ld_120, ld_128, ld_208, ld_136, ld_152 = 0, 0, 0, 0, 0, 0, 0
    ld_160, ld_168, ld_176, ld_184, ld_192, ld_200 = 0, 0, 0, 0, 0, 0

    for var in range(0, _bars - _len)[::-1]:
        p = (_ohlc.ix[var]['high'] + _ohlc.ix[var]['low'] + _ohlc.ix[var]['close']) / 3.0
        if ld_8 == 0.0:
            ld_8 = 1.0
            ld_16 = 0.0
            if (_len - 1) >= 5:
                ld_0 = _len - 1.0
            else:
                ld_0 = 5.0
            ld_80 = 100.0 * p
            ld_96 = 3.0 / (_len + 2.0)
            ld_104 = 1.0 - ld_96
        else:
            if ld_0 <= ld_8:
                ld_8 = ld_0 + 1.0
            else:
                ld_8 += 1.0
        ld_88 = ld_80
        ld_80 = 100.0 * p
        ld_32 = ld_80 - ld_88
        ld_112 = ld_104 * ld_112 + ld_96 * ld_32
        ld_120 = ld_96 * ld_112 + ld_104 * ld_120
        ld_40 = 1.5 * ld_112 - ld_120 / 2.0
        ld_128 = ld_104 * ld_128 + ld_96 * ld_40
        ld_208 = ld_96 * ld_128 + ld_104 * ld_208
        ld_48 = 1.5 * ld_128 - ld_208 / 2.0
        ld_136 = ld_104 * ld_136 + ld_96 * ld_48
        ld_152 = ld_96 * ld_136 + ld_104 * ld_152
        ld_56 = 1.5 * ld_136 - ld_152 / 2.0
        ld_160 = ld_104 * ld_160 + ld_96 * abs(ld_32)
        ld_168 = ld_96 * ld_160 + ld_104 * ld_168
        ld_64 = 1.5 * ld_160 - ld_168 / 2.0
        ld_176 = ld_104 * ld_176 + ld_96 * ld_64
        ld_184 = ld_96 * ld_176 + ld_104 * ld_184
        ld_144 = 1.5 * ld_176 - ld_184 / 2.0
        ld_192 = ld_104 * ld_192 + ld_96 * ld_144
        ld_200 = ld_96 * ld_192 + ld_104 * ld_200
        ld_72 = 1.5 * ld_192 - ld_200 / 2.0
        if ld_0 >= ld_8 and ld_80 != ld_88:
            ld_16 = 1.0
        if ld_0 == ld_8 and ld_16 == 0.0:
            ld_8 = 0.0

        if ld_0 < ld_8 and ld_72 > 0:
            ld_24 = 50.0 * (ld_56 / ld_72 + 1.0)
            if ld_24 > 100.0:
                ld_24 = 100.0
            if ld_24 < 0.0:
                ld_24 = 0.0
        else:
            ld_24 = 50.0

        line_all[var] = ld_24
        line_down[var] = ld_24
        line_up[var] = ld_24

        if line_all[var] > line_all[var + 1] - _filter:
            line_up[var] = None
        else:
            if line_all[var] < (line_all[var + 1] + _filter):
                line_down[var] = None
            else:
                if line_all[var] == (line_all[var + 1] + _filter):
                    line_down[var] = None
                    line_up[var] = None
    return line_all, line_up, line_down


def rsi(prices, period=14):
    """ RSI=100-［100/（1+RS）］= 100 * (1- 1/(1+RS)) = 100. * up / (up + abs(down))
    RS=（the days up of close / the days down of close）. usually period=14.
    :param prices:
    :param period:
    :return:
    """
    bars = len(prices)
    deltas = np.diff(prices)
    rsi_ar = np.zeros_like(prices, dtype=np.float32)

    for i in range(bars - period + 1):
        delta = pd.Series(deltas[i:i + period - 1])
        up = delta.ix[delta > 0].sum()
        down = delta.ix[delta < 0].sum()
        rsi_ar[period + i - 1] = 100. * up / (up + abs(down))

    return rsi_ar


def bollinger(prices, period=5, d=2):
    """
    :param prices:
    :param period:
    :param d:
    :return:
    """
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    up = sma + d * std
    dn = sma - d * std
    return sma, up, dn


def roc(prices, period=21):
    """ ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    Rate-of-Change  measures the percent change in price from one period to the next.
    :param prices: ndarray
    :param period: int > 1 and < len(prices) (optional and defaults to 21)
    :return: rocs ndarray
    """
    num_prices = len(prices)

    if num_prices < period:
        raise SystemExit('Error: num_prices < period')
    roc_range = num_prices - period
    rocs = np.zeros(roc_range)
    for idx in range(roc_range):
        rocs[idx] = ((prices[idx + period] - prices[idx]) / prices[idx]) * 100
    return rocs


def EMA(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def MACD(x, slow=26, fast=12):
    emaslow = EMA(x, slow)
    emafast = EMA(x, fast)
    return emaslow, emafast, emafast - emaslow


def KAMA(prices, num_er=10, num_fema=2, num_sema=30):
    """ Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)
    prices = pd.Series([107.92, 107.95])

    :return:
    """
    P = pd.DataFrame(prices, columns=['price'])

    """ Change = abs(tick_price - tick_price(10 periods ago))
    Volatility = sum(abs(tick_price - prior_tick_price), n)
    ER = Change/Volatility
    SC = [ER x (fastest SC - slowest SC) + slowest SC] ** 2
    SC = [ER x (2/(2+1) - 2/(30+1)) + 2/(30+1)] ** 2
    :return:
    """
    P['change'] = abs(prices.shift(num_er) - prices)
    P['volatility'] = abs(prices.shift(num_fema - 1) - prices)

    P['ER'] = P['change'] / P['volatility'].rolling(num_er).sum()

    P['fastestEMA'] = 2 / (num_fema + 1)
    P['slowestEMA'] = 2 / (num_sema + 1)
    P['SC'] = (P['ER'] * (P['fastestEMA'] - P['slowestEMA']) + P['slowestEMA']) ** 2
    P['KAMA'] = P['price']
    P['KAMA'][0:num_er - 1] = np.NaN

    for i in range(0, len(P)):
        if i >= num_er:
            P['KAMA'][i] = P['KAMA'][i - 1] + P['SC'][i] * (P['price'][i] - P['KAMA'][i - 1])

    return P['KAMA'].values


def StochRSI(rsi_ar, period=14):
    df_rsi = pd.DataFrame(rsi_ar, columns=['rsi'])
    df_rsi['high_rsi'] = df_rsi['rsi'].rolling(period).max()
    df_rsi['low_rsi'] = df_rsi['rsi'].rolling(period).min()
    df_rsi['stoch_rsi'] = (df_rsi.rsi - df_rsi.low_rsi) / (df_rsi.high_rsi - df_rsi.low_rsi) + 1e-6
    return df_rsi['stoch_rsi'].values


class FeatureSpace(object):
    """Small Feature Space config."""

    ohlc = ['5S', '10S', '15S', '20S', '30S', '60S', '120S', '180S', '300S']
    space_range = ['300S', '600S', '900S']
    pred_range = ['30S', '60S', '90S', '120S', '150S', '180', '']

    ''''''
    sma = ['sma5S', 'sma10S', 'sma20S', 'sma30S']
    sma_cross = ['sma5_10', 'sma10_20', 'sma20_30']

    RSI = 'RSI Momentum = close t − close t−n distance(bolling, price)'
    Stochastics_RSI = 'RSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)'

    time_feature = 'time_feature'
    Price_rate_of_change = 'Price_rate_of_change'
    New_high_breakout = 'New_high_breakout'

    DOTDetrendedPrice = 'DOTDetrendedPrice'
    Regression_fit = 'Regression_fit'
    RandomWall = 'RandomWall'
    BrownLinear = 'BrownLinear'
    HoltLinear = 'HoltLinear'


'''
dft = pd.Series([1 , 2 , 1, 2, 4, 2, 3, 5, 3, 4, 7, 7])
_asset='EURUSD'
_filename = '/home/tfl/ggprice/16-08-26 14:57:30.csv'
'''
'''
filename2 = '/home/tfl/ggprice/16-08-26 14:57:30.csv'
price_origin, price_0_100, pmax, pmin = getdf(filename2)

_, bbup, bbdown = bollinger(price_origin, period=600, d=2.5)

price_origin.plot()
bbup.plot()
bbdown.plot()'''
