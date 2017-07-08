"""
Using KNN learner to develop a trading strategy
"""

import numpy as np
import pandas as pd
from util import get_data
import datetime as dt
import matplotlib.pyplot as plt
import math


def add_bband(data, N=20, Nsd=2):
    """
    :param data: dataframe containing price info of a stock
    :param N: rolling days to compute mean and stdev
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
    bband = (data.ix[:, [symbol]] - bb_dn)/(bb_up - bb_dn)
    data['bbp'] = bband

    return data, SMA_N, stdev


def add_mmt(data, N=20):
    # times 10 to make values bigger to close to 1
    symbol = data.columns.values[1] # assuming stock price is in the second column in data
    data['mmt'] =data.ix[:,[symbol]]/data.ix[:,[symbol]].slice_shift(N-1) - 1

    return data


def add_vlt(data, N=20):

    symbol = data.columns.values[1] # assuming stock price is in the second column in data
    d_return = data.ix[:,[symbol]] / data.ix[:,[symbol]].slice_shift(1) - 1
    vlt = np.zeros((data.shape[0], 1))
    for idx in range(N - 1, data.shape[0]):
        vlt[idx] = d_return.iloc[idx - N + 1:idx + 1].std()
    data['vlt'] = vlt*1000 # make volatility values bigger closer to 1

    return data


def cal_EMA(data_series, N):
    '''
    :param data_serial: pandas series as time serial data
    :param N: looking-back window
    :return: 1-d array calculated EMA
    '''

    SMA = np.zeros((data_series.shape[0],1))
    for idx in range(N - 1, data_series.shape[0]):
        # assuming stock price is in the second column in data
        SMA[idx] = data_series.ix[idx - N + 1:idx + 1].sum() / N

    SMA[SMA == 0] = 'NaN'

    multi = 2.0 / (N + 1)
    EMA = SMA
    # EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).
    for i in range(N+1, EMA.shape[0]):
        EMA[i] = (data_series.ix[i] - EMA[i - 1]) * multi + EMA[i - 1]

    return EMA


def add_MACD(data, Ns=[12,26,9]):
    '''
    :param data: DataFrame containing stock price info in the second column
    :param Ns: List of short term long term EMA to use and look-back window of MACD's EMA
    :return:
    '''
    symbol = data.columns.values[1]  # assuming stock price is in the second column in data
    MACD = cal_EMA(data.ix[:,[symbol]],N=Ns[0]) - cal_EMA(data.ix[:,[symbol]],N=Ns[1])
    data['MACD'] = MACD
    signal = cal_EMA(data.MACD[Ns[1]:],N=Ns[2])
    # # normalized them
    # MACD = (MACD - np.nanmean(MACD))/(2*np.nanstd(MACD))
    # signal  = (signal - np.nanmean(signal))/(2*np.nanstd(signal))
    data['MACD'] = MACD
    data['Signal'] = 'NaN'
    data.loc[Ns[1]:,'Signal'] = signal

    return data


def generate_order(data):
    order = []
    date_list = data.index.values
    idx = 0
    shares = 0

    while idx < data.shape[0]: # not finished, switched to Extra point first
        if data.predY[date_list[idx]] > 0.01:
            if order == []:
                order = [[date_list[idx],data.columns.values[1],'BUY',100]]
                idx += 5
            else:
                order.append([date_list[idx],data.columns.values[1],'BUY',100])
                idx += 5
        elif data.predY[date_list[idx]] < -0.01:
            if order == []:
                order = [[date_list[idx],data.columns.values[1],'SELL',100]]
                idx += 5
            else:
                order.append([date_list[idx],data.columns.values[1],'SELL',100])
                idx += 5
        idx += 1

    orders = pd.DataFrame(order,columns=['Date', 'Symbol', 'Order', 'Shares'])
    # print orders

    return orders


def bin_divider(df_series, N=10):
    '''
    :param df_series: input in form of a pandas series
    :param N: number of bins to be divided to
    :return: a list of values used as dividing values
    '''
    dropna = df_series.dropna()
    sorted_s = dropna.values.argsort()
    jump = int(math.ceil(len(sorted_s)/N))

    return list(dropna[sorted_s[jump:-jump:jump]].values)


def get_position(indicator,divider_list):
    # find the sorted position of the indicator value in a divider_list and return as state
    list = divider_list[:]
    list.append(indicator)
    sort_all = list.sort()
    state = list.index(indicator)

    return state


def get_state(ind_values,divider_lists,holdings):
    # get 3-digit state with the order of mmt/BB%/MACD
    # the sequence of indicators and their divider lists must match
    if not len(ind_values)==len(divider_lists):
        print 'Must match indicator numbers!'

    s = []
    for i in range(0,len(ind_values)):
        s.append(get_position(ind_values[i],divider_lists[i]))

    state = 100*s[0] + 10*s[1] + s[2]

    return state

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    verbose = False
    # # initializating by reading data from files
    data_start = '2007-12-01'
    data_end = '2011-12-31'
    data = get_data(['IBM'], pd.date_range(data_start, data_end))
    data, data['SMA'], data['StDev'] = add_bband(data,N=20)
    data = add_mmt(data)
    data = add_MACD(data)
    # generate the daily return
    data['DRet'] = data.iloc[:,1].slice_shift(-1)/data.iloc[:,1] - 1

    # for testing of bollinger band only
    plt.plot(data.iloc[:,1][20:300])
    plt.plot(data.SMA[20:300]+2*data.StDev[20:300])
    plt.plot(data.SMA[20:300]-2*data.StDev[20:300])
    plt.plot(data.SMA[20:300])
    plt.show()


    # mmt_div = bin_divider(data.mmt)
    # bbp_div = bin_divider(data.bbp)
    # MACD_div = bin_divider(data.MACD)
    #
    # sd = dt.datetime(2008, 1, 1); ed = dt.datetime(2009, 1, 1)
    # temp = data.index[data.index > sd]
    # print temp[0]

    if verbose:
        print 'mmt_div:', mmt_div
        print 'bbp_div:', bbp_div
        print 'MACD_div:', MACD_div
        test = [data.mmt[500],data.bbp[500],data.MACD[500]]
        print test
        s = get_state(test,[mmt_div,bbp_div,MACD_div],1)
        print s
    # # start and end date the learner concerns
    # start_date = '2007-12-31'
    # end_date = '2009-12-31'
    # # generate training data
    # trainX = data[['bband','mmt','MACD']][start_date:end_date].values
    # trainY = data.Y[start_date:end_date].values
    #
    # plt.plot(data.iloc[:,1]['2007-12-31':'2009-12-31'],'r')
    # # adjust prediction Y to scale up with price
    # enlarge = (data.iloc[:,1].max()-data.iloc[:,1].min())/(trainY.max()-trainY.min())
    # plt.plot(data.Y[start_date:end_date]*enlarge+(data.iloc[:,1].min()+data.iloc[:,1].max())/2,'g')
    # plt.show()
    #
    # # orders = generate_order(ana_data)

if __name__ == "__main__":
    test_code()
