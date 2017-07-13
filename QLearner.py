"""
Basic structure (c) 2016 Tucker Balch
Implemented by Eric Zhang since 07/2016
Q-learner class (part of reinforcement learning algorithm)
"""

import numpy as np
import random as rand
import time
import math

class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.98,
                 radr=0.95,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar # save for checking dyna
        self.newrar = rar # create a copy of rar to use in decay
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Q_table = np.random.rand(num_states,num_actions)*2 - 1
        # self.Q_table = np.zeros((num_states,num_actions))
        self.last_s = 0 # save the last state
        self.last_a = 0 # save the last action
        self.dyna = dyna
        self.visited = []
        if dyna: # initialize matrix for dyna implementation
            self.T_ct = {} # save counts of each action at each state in a dictionary
            self.R_table = np.zeros((num_states,num_actions)) # save reward matrix as part of model


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # randomly select an action
        if rand.random > self.rar:
            action = self.Q_table[s].argmax()
        else:
            action = int(math.floor((rand.random() * self.num_actions)))
        self.rar *= self.radr  # decaying random action

        if self.verbose: print "s =", s,"a =",action
        # print self.Q_table[100]
        # save last move info
        self.last_s = s
        self.last_a = action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @param iteration: just to make sure we dont do dyna at the first query
        @returns: The selected action
        """

        is_first_iteration = self.newrar is self.rar
        # epsilon greedy select an action
        if rand.random() > self.newrar:
            action = self.Q_table[s_prime].argmax()
        else:
            action = int(math.floor((rand.random()*self.num_actions)))
        self.newrar *= self.radr # decaying random action

        # now update Q table
        self.Q_table[self.last_s, self.last_a] += \
            self.alpha*(r + self.gamma*self.Q_table[s_prime].max() - self.Q_table[self.last_s, self.last_a])

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        if self.dyna and ~is_first_iteration: # make sure we dont do dyna at the first query

            if [self.last_s,self.last_a] not in self.visited:
                self.visited.append([self.last_s,self.last_a])
                self.T_ct[(self.last_s, self.last_a)] = []
            self.T_ct[(self.last_s, self.last_a)].append(s_prime)
            self.R_table[self.last_s, self.last_a] = (1-self.alpha)*self.R_table[self.last_s, self.last_a] + self.alpha*r

            for _ in range(self.dyna):
                d_state, d_action = self.visited[int(math.floor((rand.random()*len(self.visited))))]
                d_s_prime = self.T_ct[d_state,d_action][int(math.floor((rand.random()*len(self.T_ct[(d_state,d_action)]))))]
                d_r = self.R_table[d_state,d_action]
                self.Q_table[d_state, d_action] += \
                    self.alpha * (d_r + self.gamma * self.Q_table[d_s_prime].max() - self.Q_table[d_state, d_action])

        # if self.verbose: print self.T_ct[(self.last_s, self.last_a)]

        self.last_s = s_prime
        self.last_a = action

        return action

    def output_q(self):
        return self.Q_table

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
