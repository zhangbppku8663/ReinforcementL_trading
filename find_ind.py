import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import matplotlib.pyplot as plt
import os

def test_code(verb = True):



    # set parameters for training the learner
    sym = "SPY_ghost"
    stdate = dt.datetime(2006,12,31)
    enddate = dt.datetime(2015,8,10)

    # try to record different return and states varying Nbb and Nmmt
    list = []
    for Nbb in range(18,20):
        for Nmmt in range(3,5):
    # Nbb = 20 # bollinger band looking back window
    # Nmmt = 5 # momentum looking back window
            # instantiate the strategy learner
            learner = sl.StrategyLearner(bins=10,verbose = verb)
            # train the learner
            ret, states = learner.addEvidence(symbol = sym, sd = stdate, \
                ed = enddate, sv = 10000,N_bb=Nbb,N_mmt=Nmmt,it=70, output=True)
            list.append([Nbb,Nmmt,ret,states])
            print list

if __name__=="__main__":
    test_code(verb = False)