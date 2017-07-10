"""
Basic structure (c) 2016 Tucker Balch
Implemented by Eric Zhang since 07/2016
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np
import math
import time
import os


def symbol_to_path(symbol, base_dir=os.path.join("..", "code")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


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


def bin_divider(df_series, N=10, method='quantile'):
    '''
    :param df_series: input in form of a pandas series
    :param N: number of bins to be divided to
    :param method:
            'percentile': cut series with same number of items in each bin
            'even': bins have the same width
    :return: a list of values used as dividing values
    '''
    dropna = df_series.dropna()

    if method=='quantile':
        sorted_arg = dropna.values.argsort()
        jump = int(math.ceil(len(sorted_arg)/float(N)))
        cut_offs = list(dropna[sorted_arg[jump::jump]].values)
    elif method=='even':
        sorted_s = sorted(dropna)
        cut_offs = [sorted_s[0]+(i+1)*(sorted_s[-1]-sorted_s[0])/float(N) for i in range(N-1)]
#        width = (sorted_s[-1] - sorted_s[0])/float(N)
#        cut_offs = list(np.linspace(sorted_s[0],sorted_s[-1],width))
    else: print('Warning: invalid method')

    return cut_offs


def get_position(indicator, divider_list):
    # find the sorted position of the indicator value in a divider_list and return as state
    def indsorting(ind):
        lst = divider_list[:]
        lst.append(ind)
        lst.sort()
        pos = lst.index(ind)
        return pos
    poses = indicator.apply(indsorting)

    return poses


def get_state(ind_values, divider_dict):
    # get state given the space of indicators
    # can deal with either DataFrame input (many observations) or Series (just one)
    # the sequence of indicators is determined by the dataframe columns or series index
    if isinstance(ind_values, pd.DataFrame):
        inds = list(ind_values.columns)
        if not len(ind_values.columns)==len(divider_dict): # check length
            print('Warning: Must match indicator numbers!')
        s = np.zeros([len(divider_dict), len(ind_values)]) # notice the column number is the len(ind_values)
        state = np.zeros([1,len(ind_values)])
        for indicator, i in zip(inds, range(len(divider_dict))):
            s[i] = get_position(ind_values[indicator],divider_dict[indicator])
            state = state + s[i]*10**i
    elif isinstance(ind_values, pd.Series):
        inds = list(ind_values.index)
        if not len(ind_values)==len(divider_dict):
            print('Warning: Must match indicator numbers!')
        s = np.zeros([len(divider_dict),1])
        state = 0
        for indicator, i in zip(inds, range(len(divider_dict))):
            s[i] = get_position(pd.Series(ind_values[indicator]),divider_dict[indicator])
            state = state + s[i]*10**i
    else: print('Warning: we need pandas DataFrame or Series as inputs')

    return state.astype('int').flatten()


def full_state(ind_states, holdings, num_of_ind=3):
    # generate states considering holdings

    return int((holdings+100)*10**(num_of_ind-2)+ind_states)


def set_sprime(data, date_ind, holdings, action):
    # action 0 SELL, 1 NOTHING, 2 BUY
    if holdings == 100:
        if action==2:
            pass
            # print 'Not allowed holdings'
        elif action==1:
            pass
        elif action==0:
            holdings -= 100

    elif holdings == 0:
        if action==2:
            holdings += 100
        elif action==1:
            pass
        elif action==0:
            holdings -= 100

    elif holdings == -100:
        if action==2:
            holdings += 100
        elif action==1:
            pass
        elif action==0:
            pass
            # print 'Not allowed holdings'
    s_prime = int(data.ix[date_ind,'Ind_States'] + 10*(holdings+100))

    return s_prime, holdings


class StrategyLearner(object):

    # constructor
    def __init__(self, bins=10, verbose=False):
        self.verbose = verbose
        self.div_dict = {}
        self.bins = bins
        self.states_log = []
        self.test_log = []
        self.total_states = []

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008,1,1),
                    ed=dt.datetime(2009,1,1),
                    sv=10000,
                    N_mmt=20,
                    N_bb=20,
                    it=50,
                    output=False):

        # we need to read from the earliest possible date to get the same EMA for indicator calculations
        base_dir = os.path.join("..", "data")
        path = os.path.join(base_dir, "{}.csv".format(str(symbol)))
        df_temp = pd.read_csv(path, parse_dates=True, index_col='Date',
                              usecols=['Date', 'Adj Close'], na_values=['nan'])
        data_sd = sorted(df_temp.index)[0]

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(data_sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        if self.verbose: print prices

        # calculate technical indicators
        data = add_bband(prices_all, N=N_bb)
        data = add_mmt(data, N=N_mmt)
        data = add_MACD(data)
        # data['MACD_sig'] = data.MACD - data.Signal

        indicators = ['mmt',
                      'bbp',
                      'MACD']
        # find the true training starting day when market opened (data_sd is earlier than this day)
#        temp = data.index[data.index >= sd]
#        sd = temp[0]
        sd = data.index[data.index >= sd][0]

        # create a dataframe within sd and ed
        df = data[sd:].copy()
        # generate dividing values to bin indicators
        for ind in indicators:
            self.div_dict[ind] = bin_divider(df[ind],N=self.bins, method='even')

        # without considering holdings, get partial states based on indicators all at once
        ind_states = get_state(df[['mmt','bbp','MACD']],self.div_dict)
        df['Ind_States'] = ind_states

        self.learner = ql.QLearner(num_states=3000,num_actions=3,dyna=200,rar=0.9999,radr=0.9999,gamma=0.99,verbose=False)

        bestr = 0.0
        for iteration in range(0,it):

            holdings = 0
            start_state = full_state(df.loc[sd,'Ind_States'], holdings, num_of_ind=len(indicators))
            action = self.learner.querysetstate(start_state)
            # log in states that has been in
            if start_state not in self.states_log:
                self.states_log.append(start_state)

            i = 1 # use to get one day ahead of current date
            cash = sv; portv_old = sv
            for date in df.index[:-1]:

                # first check if action is allowed
                if action==0 and holdings==-100:
                    action = 1
                elif action==2 and holdings==100:
                    action = 1

                # set the s prime state and holdings according to action
                newstate, newhold = set_sprime(df,i,holdings,action)

                if newstate not in self.states_log:
                    self.states_log.append(newstate)

                # calculate portfolio values. portv = cash + stock
                cash -= df.ix[date,syms].values*(newhold-holdings)
                portv = cash + df.ix[i,syms].values*newhold
                # calculate the reward as daily portfolio return
                # big number to wash out random initiation of Q-table
                rwd = 100*(portv - portv_old)/portv
                # query for the next action
                action = self.learner.query(newstate,rwd,iteration)

                holdings = newhold
                portv_old = portv
                i += 1

            print 'CumRet is:', float(portv)/sv - 1
            if output == True and iteration > 50:  # record the best portv after 50 iterations
                if float(portv)/sv - 1 > bestr:
                    bestr = float(portv)/sv - 1
            print 'Explored states:', len(self.states_log)
            print iteration, ':', i, 'days of trading'

        self.total_states = list(set(df.Ind_States.values))
        print "We have", len(self.total_states),"states initially not considering holdings."
        print "Meaning possible states are", len(self.total_states)*3
        # if need the information of cum. ret. and divided states, return them
        if output:
            return bestr

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM",
                   sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1),
                   sv = 10000, N_mmt=20, N_bb=20):

        # we need to read from the earliest possible date to get the same EMA for indicator calculations
        base_dir = os.path.join("..", "data")
        path = os.path.join(base_dir, "{}.csv".format(str(symbol)))
        df_temp = pd.read_csv(path, parse_dates=True,index_col='Date', usecols=['Date','Adj Close'], na_values=['nan'])
        data_sd = sorted(df_temp.index)[0]

        syms = [symbol]
        dates = pd.date_range(data_sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY

        # trades_SPY = prices_all['SPY_test']  # only SPY, for comparison later

        # add on indicators
        data = add_bband(prices_all,N=N_bb)
        data = add_mmt(data,N=N_mmt)
        data = add_MACD(data)
        data['MACD_sig'] = data.MACD - data.Signal

        # find the true training starting day when market opens (data_sd is earlier than this day)
        temp = data.index[data.index>=sd]
        sd = temp[0]

        # create a dataframe within sd and ed
        df = data[sd:].copy()

        # without considering holdings, get partial states based on indicators all at once
#        ind_states = np.zeros((df.shape[0],1),dtype=int)
#        for id in range(0,df.shape[0]):
#            ind_states[id] = get_state([df.mmt[id],df.bbp[id],df.MACD_sig[id]],[self.mmt_div,self.bbp_div,self.MACD_div])
        ind_states = get_state(df[['mmt','bbp','MACD']],self.div_dict)        
        df['Ind_States'] = ind_states

        trades = prices_all[[symbol,]][sd:]  # only portfolio symbols from start date to end date
        trades.values[:, :] = 0  # set them all to nothing
        holdings = 0
        start_state = full_state(df.loc[sd, 'Ind_States'], holdings)
        if start_state not in self.test_log:
            self.test_log.append(start_state)
        action = self.learner.querysetstate(start_state)
        i = 1
        cash = sv
        for date in df.index[:-1]:

            # first check if action is allowed
            if action == 0 and holdings == -100:
                action = 1
            elif action == 2 and holdings == 100:
                action = 1

            # set the s prime state according to action
            newstate, newhold = set_sprime(df, i, holdings, action)
            # record states
            if newstate not in self.test_log:
                self.test_log.append(newstate)

            # record trades
            if not newhold==holdings:
                if action == 2:
                    trades.values[i-1,:] = 100
                elif action == 0:
                    trades.values[i-1,:] = -100

            # calculate portfolio values. portv = cash + stock
            cash -= int(df.ix[date, syms].values * (newhold - holdings))
            portv = cash + int(df.ix[i, syms].values * newhold)
            # query for the next action
            action = self.learner.querysetstate(newstate)
            holdings = newhold
            i += 1

        print float(portv) / sv - 1

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all

        # now check what states have not been explored before
        not_seen = []; test_ind_states = []
        for item in self.test_log:
            test_ind_states.append(int(item%1000))
        possible_states = list(set(test_ind_states))


        print 'Test session hit', len(possible_states),'states.'
        # get unseen states in test process not considering holdings
        for item in df.Ind_States:
            if item not in self.total_states:
                not_seen.append(item)
        print len(not_seen), "indicator states not seen before in training:",not_seen

        self.test_log.sort()
        # fill zeros at the end of testsstates list in order to make the same length
        for _ in range(0,len(self.states_log)-len(self.test_log)):
            self.test_log.append(0)

        self.states_log.sort()
        dict = {'Learned':self.states_log,'Tested':self.test_log}
        states = pd.DataFrame(dict)
        states.to_csv(symbol_to_path('States'))

        return trades

    def output(self):
        q_table = self.learner.output_q()
        return q_table, self.div_dict

if __name__=="__main__":
    print "One does not simply think up a strategy"

