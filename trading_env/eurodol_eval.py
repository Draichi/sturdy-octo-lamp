import random
import pandas as pd
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
dt = pd.read_csv("EURUSD_5.csv", header=None, usecols=range(0,67), skiprows=39, engine='python')

class EuroDol(Env):
    """docstring for EuroDol."""
    def __init__(self, filename, seq_len=200):
        super(EuroDol, self).__init__()
        self.data = dt[-1:]
        print('data',self.data)
        observation_space = (len(self.data.iloc[0]),)
        self.seq_len = seq_len
        self.current_start = None
        self.current_offset = None
        self.action_space = Discrete(4)  # buy, sell, hold, close all
        self.observation_space = observation_space
        self.current_buys = []
        self.current_sells = []

    def step(self, action):
        # only 1 step
        # state = self.data.iloc[self.current_start + self.current_offset + 1].values
        state = self.data.iloc[0].values
        done = True
        if action == 0:    # buy
            reward = 0
            self.current_buys.append(state[3])
            print('buy',state[3])
        elif action == 1:  # sell
            reward = 0
            self.current_sells.append(state[3])
            print('sell',state[3])
        elif action == 2:  # hold
            reward = 0
            print('hold')
        elif action == 3:  # close all
            close_price = state[3]
            buys = np.array(self.current_buys)
            sells = np.array(self.current_sells)
            reward = (close_price - buys).sum() + (sells - close_price).sum()
            print('close all - reward:',reward,'close_price:',close_price)

        return state, done, reward, None

    def reset(self):
        # self.current_start = random.randint(0, len(self.data))
        # self.current_offset = 0
        return self.data.iloc[0].values

    def render(self):
        # print("State :", self.data.iloc[self.current_start + self.current_offset])
        # print(self.action_space)
        pass
