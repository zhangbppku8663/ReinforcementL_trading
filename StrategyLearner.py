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
from util import symbol_to_path, add_bband, add_MACD, add_mmt


class StrategyLearner(object):

    # constructor
    def __init__(self, bins=10, div_method='even', indicators=None, verbose=False):
        self.verbose = verbose
        self.div_dict = {}
        self.bins = bins
        self.div_method = div_method
        self.states_log = []
        self.test_log = []
        self.evi_states = []
        self.learner = ql.QLearner()
        self.df = pd.DataFrame()  # for training
        self.tdf = pd.DataFrame()  # for test
        if indicators is None:
            self.indicators = ['mmt', 'bbp', 'MACD']
        else:
            self.indicators = indicators

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
        path = symbol_to_path(symbol=symbol)
        df_temp = pd.read_csv(path, parse_dates=True, index_col='Date',
                              usecols=['Date', 'Adj Close'], na_values=['nan'])
        data_sd = sorted(df_temp.index)[0]

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(data_sd, ed)
        data = ut.get_data(syms, dates)  # automatically adds SPY
        prices = data[syms]  # only portfolio symbols

        if self.verbose: print prices

        # calculate technical indicators
        for ind in self.indicators:
            if ind == 'bbp':
                data = add_bband(data, N=N_bb)
            elif ind == 'mmt':
                data = add_mmt(data, N=N_mmt)
            elif ind == 'MACD':
                data = add_MACD(data)
            else: print('Method to calculate not implemented!')
        # data['MACD_sig'] = data.MACD - data.Signal

        # find the true training starting day when market opened (data_sd is earlier than this day)
        sd = data.index[data.index >= sd][0]

        # create a dataframe within sd and ed
        self.df = data[sd:ed].copy()
        # create a Series of dates for convenience
        date_index = pd.Series(self.df.index)
        # generate dividing values to bin indicators
        for ind in self.indicators:
            self.div_dict[ind] = self._bin_divider(indicator=ind, method=self.div_method)

        # without considering holdings, get partial states based on indicators all at once
        self.df['Ind_States'] = self._get_state(self.df[self.indicators])

        self.learner = ql.QLearner(num_states=3000,
                                   num_actions=3,
                                   dyna=200,
                                   rar=0.999,
                                   radr=0.999,
                                   gamma=0.9,
                                   verbose=False)

        bestr = 0.0
        for iteration in range(0,it):

            holdings = 0
            start_state = self._full_state(self.df.loc[sd,'Ind_States'], holdings)
            action = self.learner.querysetstate(start_state)
            # log in states that has been in
            if start_state not in self.states_log:
                self.states_log.append(start_state)

            next_day = self.df.index[1] # use to get one day ahead of current date
            cash = sv; portv_old = sv
            for date in self.df.index[:-1]:

                # first check if action is allowed
                if action == 0 and holdings == -100:
                    action = 1
                elif action == 2 and holdings == 100:
                    action = 1

                # set the s prime state and holdings according to action
                newstate, newhold = self.set_sprime(self.df,next_day,holdings,action)

                if newstate not in self.states_log:
                    self.states_log.append(newstate)

                # calculate portfolio values. portv = cash + stock
                cash -= self.df.loc[date,syms].values*(newhold-holdings)
                portv = cash + self.df.loc[next_day,syms].values*newhold
                # calculate the reward as daily portfolio return
                # big number to wash out random initiation of Q-table
                rwd = 100*(portv - portv_old)/portv
                # query for the next action
                action = self.learner.query(newstate, rwd, iteration)

                holdings = newhold
                portv_old = portv
                today_date = list(date_index).index(date)
                next_day = date_index[today_date+1]

            print 'CumRet is:', float(portv)/sv - 1
            if output == True and iteration > 50:  # record the best portv after 50 iterations
                if float(portv)/sv - 1 > bestr:
                    bestr = float(portv)/sv - 1
            print 'Explored states:', len(self.states_log)
            print iteration, ':', len(date_index), 'days of trading'

        self.evi_states = list(set(self.df.Ind_States.values))
        print "We have", len(self.evi_states),"states initially not considering holdings."
        print "Meaning possible states are", len(self.evi_states)*3
        # if need the information of cum. ret. and divided states, return them
        if output:
            return bestr

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM",
                   sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1),
                   sv = 10000, N_mmt=20, N_bb=20):

        # we need to read from the earliest possible date to get the same EMA for indicator calculations
        path = symbol_to_path(symbol=symbol)
        df_temp = pd.read_csv(path, parse_dates=True,index_col='Date', usecols=['Date','Adj Close'], na_values=['nan'])
        data_sd = sorted(df_temp.index)[0]

        syms = [symbol]
        dates = pd.date_range(data_sd, ed)
        data = ut.get_data(syms, dates)  # automatically adds SPY

        # trades_SPY = prices_all['SPY_test']  # only SPY, for comparison later

        # add on indicators
        for ind in self.indicators:
            if ind == 'bbp':
                data = add_bband(data, N=N_bb)
            elif ind == 'mmt':
                data = add_mmt(data, N=N_mmt)
            elif ind == 'MACD':
                data = add_MACD(data)
            else: print('Method to calculate not implemented!')
        # data['MACD_sig'] = data.MACD - data.Signal

        # find the true training starting day when market opens (data_sd is earlier than this day)
        temp = data.index[data.index>=sd]
        sd = temp[0]

        # create a dataframe within sd and ed
        self.tdf = data[sd:ed].copy()
        # create a Series of dates for convenience
        date_index = pd.Series(self.tdf.index)
        # without considering holdings, get partial states based on indicators all at once
        self.tdf['Ind_States'] = self._get_state(self.tdf[self.indicators])

        # trades = data[[symbol,]][sd:].copy()  # only portfolio symbols from start date to end date
        trades = data.loc[sd:, symbol]
        trades[:] = 0  # set them all to nothing
        holdings = 0
        start_state = self._full_state(self.tdf.loc[sd, 'Ind_States'], holdings)
        if start_state not in self.test_log:
            self.test_log.append(start_state)
        action = self.learner.querysetstate(start_state)
        next_day = self.tdf.index[1]; portv = 0
        cash = sv
        for date in self.tdf.index[:-1]:

            # first check if action is allowed
            if action == 0 and holdings == -100:
                action = 1
            elif action == 2 and holdings == 100:
                action = 1

            # set the s prime state according to action
            newstate, newhold = self.set_sprime(self.tdf, next_day, holdings, action)
            # record states
            if newstate not in self.test_log:
                self.test_log.append(newstate)

            # record trades
            if not newhold==holdings:
                if action == 2:
                    trades[date] = 100
                elif action == 0:
                    trades[date] = -100

            # calculate portfolio values. portv = cash today + stock value today
            cash -= int(self.tdf.loc[date, syms].values * (newhold - holdings))
            portv = cash + int(self.tdf.loc[date, syms].values * newhold)
            # query for the next action
            action = self.learner.querysetstate(newstate)
            holdings = newhold
            today_index = list(date_index).index(date)
            next_day = date_index[today_index+1]

        print float(portv) / sv - 1

        if self.verbose: print type(trades) # it better be a Series!
        if self.verbose: print trades
        # if self.verbose: print prices_all

        # now check what states have not been explored before
        not_seen = []; test_ind_states = []
        for item in self.test_log:
            test_ind_states.append(int(item%1000))
        possible_states = list(set(test_ind_states))

        print 'Test session hit', len(possible_states),'states.'
        # get unseen states in test process not considering holdings
        for item in self.tdf.Ind_States:
            if item not in self.df.Ind_States.values:
                not_seen.append(item)
        print len(set(not_seen)), "indicator states not seen before in training:",not_seen

        self.test_log.sort()
        # fill zeros at the end of testsstates list in order to make the same length
        for _ in range(0,len(self.states_log)-len(self.test_log)):
            self.test_log.append(0)

        self.states_log.sort()
        logs = {'Learned':self.states_log,'Tested':self.test_log}
        states = pd.DataFrame(logs)
        states.to_csv(symbol_to_path('States'), index=False)

        return trades

    def output(self):
        q_table = self.learner.output_q()
        return q_table, self.div_dict

    @staticmethod
    def set_sprime(data, date_ind, holdings, action):
        # action 0 SELL, 1 NOTHING, 2 BUY
        if holdings == 100:
            if action == 2:
                pass
                # print 'Not allowed holdings'
            elif action == 1:
                pass
            elif action == 0:
                holdings -= 100

        elif holdings == 0:
            if action == 2:
                holdings += 100
            elif action == 1:
                pass
            elif action == 0:
                holdings -= 100

        elif holdings == -100:
            if action == 2:
                holdings += 100
            elif action == 1:
                pass
            elif action == 0:
                pass
                # print 'Not allowed holdings'
        s_prime = int(data.loc[date_ind, 'Ind_States'] + 10 * (holdings + 100))

        return s_prime, holdings

    def _bin_divider(self, indicator, method='even'):
        '''
        :param indicator: related indicator name as string
        :param method:
                'quantile': cut series with same number of items in each bin
                'even': bins have the same width
        :return: a list of values used as dividing values
        '''
        dropna = self.df[indicator].dropna()

        if method == 'quantile':
            sorted_arg = dropna.values.argsort()
            jump = int(math.ceil(len(sorted_arg) / float(self.bins)))
            cut_offs = list(dropna[sorted_arg[jump::jump]].values)
        elif method == 'even':
            sorted_s = sorted(dropna)
            cut_offs = [sorted_s[0] + (i + 1) * (sorted_s[-1] - sorted_s[0]) / float(self.bins) for i in range(self.bins - 1)]
        else:
            cut_offs = None
            print('Warning: invalid method')

        return cut_offs

    def _get_position(self, ind_values):
        # find the sorted position of the indicator value in a divider_list and return as state
        indicator = ind_values.name
        def ind_sorting(ind):
            lst = self.div_dict[indicator][:]
            lst.append(ind)
            lst.sort()
            pos = lst.index(ind)
            return pos

        poses = ind_values.apply(ind_sorting)

        return poses

    def _get_state(self, inds_values):
        # get state given the space of indicators
        # can deal with either DataFrame input (many observations) or Series (just one)
        # the sequence of indicators is determined by the dataframe columns or series index
        if isinstance(inds_values, pd.DataFrame):
            indicators = list(inds_values.columns)
            s = np.zeros([len(self.div_dict), len(inds_values)])  # notice the column number is the len(ind_values)
            state = np.zeros([1, len(inds_values)])
            for indicator, i in zip(indicators, range(len(self.div_dict))):
                s[i] = self._get_position(inds_values[indicator])
                state = state + s[i] * 10 ** i
        elif isinstance(inds_values, pd.Series):
            indicators = list(inds_values.index)
            s = np.zeros([len(self.div_dict), 1])
            state = 0
            for indicator, i in zip(indicators, range(len(self.div_dict))):
                s[i] = self._get_position(pd.Series(inds_values[indicator]))
                state = state + s[i] * 10 ** i
        else:
            state = 0
            print('Warning: we need pandas DataFrame or Series as inputs')

        return state.astype('int').flatten()

    def _full_state(self, ind_states, holdings):
        # generate states considering holdings
        state = int((holdings + 100) * 10 ** (len(self.indicators) - 2) + ind_states)
        return state


if __name__=="__main__":
    print "One does not simply think up a strategy"

