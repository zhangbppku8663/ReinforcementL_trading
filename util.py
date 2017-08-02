"""all utility codes"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def symbol_to_path(symbol, base_dir=os.path.join(os.getcwd(),"data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY_test' not in symbols:  # add SPY_test for reference, if absent
        symbols = ['SPY_test'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY_test':  # drop dates SPY_test did not trade
            df = df.dropna(subset=["SPY_test"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def add_bband(data, N=20, Nsd=2):
    """
    :param data: dataframe containing price info of a stock
    :param N: rolling days to compute mean and stdev
    :param Nsd: how many times of stdev to add to deduct
    :return: a data frame with percentage of bollinger band of price
            calculated after first N days
    """
    symbol = data.columns.values[1]  # assuming stock price is in the second column in data
    SMA_N = np.zeros((data.shape[0], 1))  # save the N day average price
    stdev = np.zeros((data.shape[0], 1))  # save the  N day standard deviation
    for idx in range(N - 1, data.shape[0]):
        SMA_N[idx] = data.ix[idx - N + 1:idx + 1, [symbol]].sum() / N
        stdev[idx] = data.ix[idx - N + 1:idx + 1, [symbol]].std()
    SMA_N[SMA_N == 0] = 'NaN'  # reset zero valuses to NaN for proper plot

    # Calculate upper and lower bands
    bb_up = SMA_N + Nsd * stdev
    bb_dn = SMA_N - Nsd * stdev
    bb_up[bb_up == 0] = 'NaN'  # reset zero valuses to NaN for proper plot
    bb_dn[bb_dn == 0] = 'NaN'  # reset zero valuses to NaN for proper plot

    # calculate distance of current price to upper and lower bollinger band, respectively
    bband = (data.ix[:, [symbol]] - bb_dn) / (bb_up - bb_dn)
    data['bbp'] = bband

    return data


def add_mmt(data, N=20):

    symbol = data.columns.values[1]  # assuming stock price is in the second column in data
    data['mmt'] = data.ix[:, [symbol]] / data.ix[:, [symbol]].slice_shift(N - 1) - 1

    return data


def cal_EMA(data_series, N):
    '''
    :param data_series: pandas series as time serial data
    :param N: looking-back window
    :return: 1-d array calculated EMA
    '''

    SMA = np.zeros((data_series.shape[0], 1))

    for idx in range(N - 1, data_series.shape[0]):
        # assuming stock price is in the second column in data
        SMA[idx] = data_series.ix[idx - N + 1:idx + 1].sum() / N

    SMA[0:N] = 'NaN'

    multi = 2.0 / (N + 1)
    EMA = SMA
    # EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).
    for i in range(N + 1, EMA.shape[0]):
        EMA[i] = (data_series.ix[i] - EMA[i - 1]) * multi + EMA[i - 1]

    return EMA

def add_ATR(data, N=14):
    symbol = [data.columns.values[1]]  # must be an item of a list, because get_data iterate on it
    index = data.index  # get all dates that market opens
    high = get_data(symbols=symbol, dates=index, colname='High')
    low = get_data(symbols=symbol, dates=index, colname='Low')
    T_range = pd.DataFrame(index=index)
    T_range['TR1'] = high[symbol] - low[symbol] # today's high subtracted by today's low
    T_range['TR2'] = high[symbol] - data[symbol].shift(1)  # today's high subtracted by yesterday's close
    T_range['TR3'] = data[symbol].shift(1) - low[symbol]  # yesterday's close subtracted by today's low
    max_TR = T_range.max(axis=1)
    data['ATR'] = cal_EMA(max_TR, N)

    return data

def add_MACD(data, Ns=None):
    '''
    :param data: DataFrame containing stock price info in the second column
    :param Ns: List of short term long term EMA to use and look-back window of MACD's EMA
    :return:
    '''
    if Ns is None:
        Ns = [12, 26, 9]
    symbol = data.columns.values[1]  # assuming stock price is in the second column in data
    MACD = cal_EMA(data.loc[:, symbol], N=Ns[0]) - cal_EMA(data.loc[:, symbol], N=Ns[1])
    data['MACD'] = MACD
    signal = cal_EMA(data.MACD[Ns[1]:], N=Ns[2])
    # # normalized them
    # MACD = (MACD - np.nanmean(MACD))/(2*np.nanstd(MACD))
    # signal  = (signal - np.nanmean(signal))/(2*np.nanstd(signal))
    # data['MACD'] = MACD
    data['Signal'] = 'NaN'
    data.loc[Ns[1]:, 'Signal'] = signal

    return data