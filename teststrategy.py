"""
Basic structure (c) 2016 Tucker Balch
Implemented by Eric Zhang since 07/2016
test code for the trading strategy using reinforcement learning
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import matplotlib.pyplot as plt
import numpy as np
from util import symbol_to_path
import os

import random
random.seed(42)

def generate_order(data):
    '''
    :param data: DataFrame containing all necessary signal data
    :return: a DataFrame with all order info; date as lists of all kinds of orders
            A 'order.csv' file is also generated
    '''
    order = []; idx = 0; shares = 0
    date_list = data.index.values
    symbol = data.name # assume the name of Series stores symbol info
    l_en = []; s_en = []; ext = []
    while idx < data.shape[0]:

        if data.values[idx]==100 and shares == 0:
            order.append([date_list[idx], symbol, 'BUY', 100])
            shares += 100
            l_en.append(date_list[idx])
            idx += 1
        elif data.values[idx]==-100 and shares == 100:
            order.append([date_list[idx], symbol, 'SELL', 100])
            shares -= 100
            ext.append(date_list[idx])
            idx += 1
        elif data.values[idx]==-100 and shares == 0:
            order.append([date_list[idx], symbol, 'SELL', 100])
            shares -= 100
            s_en.append(date_list[idx])
            idx += 1
        elif data.values[idx]==100 and shares == -100: # > data.predY[date_list[idx-1]]
            order.append([date_list[idx], symbol, 'BUY', 100])
            shares += 100
            ext.append(date_list[idx])
            idx += 1
        else:
            idx += 1

    orders = pd.DataFrame(order,columns=['Date', 'Symbol', 'Order', 'Shares'])

    return orders, l_en, s_en, ext


def compute_portvals(orders_file, start_val=1000000, lvrg=False):
    '''
    :param orders_file: path and name of .csv order file
    :param start_val: starting value of the portfolio
    :param lvrg: whether to consider leverage of the portfolio
    :return: date-to-date portfolio values; start and end date of orders
    '''

    # read orders, save in a DataFrame and sort according to Date
    # orders = pd.read_csv(orders_file, parse_dates=True, usecols=['Date', 'Symbol','Order','Shares'], na_values=['nan'])
    orders = orders_file.sort_values(['Date'])
    start_date, end_date = orders['Date'][0], orders['Date'].max()

    symbols = list(set(orders.Symbol))
    # prtf = pd.DataFrame(index=pd.date_range(start_date, end_date))
    # symb_vals = ut.get_data(symbols, pd.date_range(start_date, end_date))
    # prtf = prtf.join(symb_vals)
    # prtf = prtf.dropna(subset=["SPY_test"])
    prtf = ut.get_data(symbols, pd.date_range(start_date, end_date))
    for symbol in symbols:
        prtf[symbol+'_shares'] = 0

    prtf['Cash'] = start_val
    prtf['Value'] = 0
    prtf_old = prtf.copy() # save portfolio before change
    lev_old = 0 # save leverage from last time
    for idx in range(0,orders.shape[0]):
        # save original portfolio before process order
        if idx > 0:
            if not orders.Date[idx] == orders.Date[idx-1]:
                prtf_old = prtf.copy()

        # process orders following time sequence
        if orders.Order[idx] == 'BUY':
            prtf.loc[orders.Date[idx]:, orders.Symbol[idx] + '_shares'] += orders.Shares[idx]
            prtf.loc[orders.Date[idx]:,'Cash'] -= \
                prtf.loc[orders.Date[idx],orders.Symbol[idx]]*orders.Shares[idx]
        else:
            prtf.loc[orders.Date[idx]:, orders.Symbol[idx] + '_shares'] -= orders.Shares[idx]
            prtf.loc[orders.Date[idx]:,'Cash'] += \
                prtf.loc[orders.Date[idx],orders.Symbol[idx]]*orders.Shares[idx]

        if lvrg:
            # now check the leverage
            longs = 0; shorts = 0
            for symbol in symbols:
                if prtf.loc[orders.Date[idx], symbol+'_shares'] > 0:
                    longs += prtf.loc[orders.Date[idx], symbol+'_shares']*prtf.loc[orders.Date[idx], symbol]
                elif prtf.loc[orders.Date[idx], symbol+'_shares'] < 0:
                    shorts += prtf.loc[orders.Date[idx], symbol+'_shares']*prtf.loc[orders.Date[idx], symbol]
            leverage = (longs-shorts)/(longs+shorts+prtf.loc[orders.Date[idx], 'Cash'])
            if leverage > 2.0 and leverage >= lev_old:
                prtf = prtf_old
            else:
                lev_old = leverage

    for symbol in symbols:
        prtf.Value = prtf.Value + prtf.loc[:, symbol+'_shares']*prtf.loc[:, symbol]
    prtf.Value = prtf.Value + prtf.Cash
    portvals = prtf.Value

    return portvals, start_date, end_date


def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Get daily portfolio value
    cr = port_val[-1]/port_val[0] - 1
    daily_return = port_val[1:].values/port_val[0:-1].values-1
    adr = daily_return.mean()
    sddr = daily_return.std(ddof=1)
    sr = sf**(1/2.0) * (adr - rfr)/sddr

    return cr, adr, sddr, sr


def print_port(of, sv=1000000, output=False, lvrg=False, symbol='SPY'):
    '''
    :param of: .csv order file
    :param sv: starting value of the portfolio
    :param output: whether to output info
    :param lvrg: whether to consider leverage of the portfolio
    :param symbol: stock symbol to consider
    :return: void
    '''

    portvals, start_date, end_date = compute_portvals(orders_file=of, start_val=sv, lvrg=lvrg)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)
    # SPX data originally has a'$" in front which creates problem as an improper name
    symb_vals = ut.get_data([symbol], pd.date_range(start_date, end_date))

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(symb_vals.SPY_test)

    if output:
        # Compare portfolio against $SPX
        print "Date Range: {} to {}".format(start_date, end_date)
        print
        print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
        print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
        print
        print "Cumulative Return of Fund: {}".format(cum_ret)
        print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
        print
        print "Standard Deviation of Fund: {}".format(std_daily_ret)
        print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
        print
        print "Average Daily Return of Fund: {}".format(avg_daily_ret)
        print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
        print
        print "Final Portfolio Value: {}".format(portvals[-1])

        norm_port = portvals / portvals[0]
        norm_SPY = symb_vals.SPY_test / symb_vals.SPY_test[0]
        plt.plot(norm_port)
        plt.plot(norm_SPY)
        plt.title('Daily portfolio value and SPY')
        plt.ylabel('Normalized price')
        plt.legend(['Portfolio', 'SPY'], loc=2)
        plt.show()


def test_code(verb=True):
    # instantiate the strategy learner
    learner = sl.StrategyLearner(bins=10, div_method='quantile', indicators=['ATR', 'bbp'], verbose=verb)
    # set parameters for training the learner
    sym = "VIX"
    money = 2000
    stdate = dt.datetime(2013,1,1)
    enddate = dt.datetime(2016,7,31)
    Nbb = 24  # bollinger band looking back window
    Nmmt = 3  # momentum looking back window
    # train the learner
    bestr = learner.addEvidence(symbol=sym, sd=stdate,
                                ed=enddate, sv=money,
                                N_bb=Nbb, N_mmt=Nmmt,
                                it=70, output=True)
    print('Best return is', bestr)
    print('Now rar is:', learner.learner.newrar)
    #############
    # Test
    #############
    st_date = dt.datetime(2016, 8, 1)
    en_date = dt.datetime(2017, 8, 1)

    syms = [sym]
    dates = pd.date_range(st_date, en_date)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices
    #
    # test the learner
    df_trades = learner.testPolicy(symbol=sym, sd=st_date, ed=en_date,
                                   sv=money, N_bb=Nbb, N_mmt=Nmmt)
    #
    # a few sanity checks
    # df_trades should be a series)
    # including only the values 100, 0, -100
    if type(df_trades) is not pd.core.series.Series:
        print "Returned result is not a Series"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck <= 100] = 0
    tradecheck[tradecheck > 0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violates holding restrictions (more than 100 shares)"

    if verb: print df_trades

    # generate orders based on principle
    orders, l_en, s_en, ext = generate_order(df_trades)
    orders.to_csv(symbol_to_path('orders', base_dir=os.getcwd()))
    # for plot
    plt.plot(prices)
    for i in l_en:
        plt.axvline(x=i, color='g')
    for i in s_en:
        plt.axvline(x=i, color='r')
    for i in ext:
        plt.axvline(x=i, color='k')
    plt.show()

    # feed orders to the market simulator and print back-testing outputs
    print_port(of=orders, sv=money, output=True, lvrg=False, symbol=sym)
    # output Q table and indicator divider info
    q, dividers = learner.output()
    q_table = pd.DataFrame(q)

    ind_dividers = pd.DataFrame(dividers)
    q_table.to_csv(symbol_to_path('Q_Table', base_dir=os.getcwd()))
    ind_dividers.to_csv(symbol_to_path('Dividers', base_dir=os.getcwd()))

if __name__=="__main__":
    test_code(verb=False)
