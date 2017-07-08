"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.newrar = rar # create a copy of rar to use in decay
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Q_table = np.random.random_sample((num_states,num_actions)) - 0.5
        self.last_s = 0 # save the last state
        self.last_a = 0 # save the last action

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # action = rand.randint(0, self.num_actions-1)
        # randomly select an action
        if rand.random > self.newrar:
            action = self.Q_table[s].argmax()
        else:
            action = rand.randint(0, self.num_actions - 1)
        self.newrar *= self.radr  # decaying random action

        if self.verbose: print "s =", s,"a =",action

        # save last move info
        self.last_s = s
        self.last_a = action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # action = rand.randint(0, self.num_actions-1)

        # now update Q table
        max_Q = self.Q_table[s_prime].max()
        self.Q_table[self.last_s, self.last_a] += \
            self.alpha*(r + self.gamma*max_Q - self.Q_table[self.last_s, self.last_a])

        # randomly select an action
        if rand.random() > self.newrar:
            action = self.Q_table[s_prime].argmax()
        else:
            action = rand.randint(0, self.num_actions - 1)
        self.newrar *= self.radr # decaying random action

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r, self.Q_table[s_prime]
        # save last move info
        self.last_s = s_prime
        self.last_a = action

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
