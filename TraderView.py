import gym 
from gym.utils import seeding 
import numpy as np
from pathlib import Path
import math 

#actions constant
LONG = 0
SHORT = 1
FLAT = 2
BUY = 0
SELL = 1
HOLD = 2

#using the OPEN, HIGH, LOW, CLOSE and VOLUME chart
class OhlcvEnv(gym.Env):

    def __init__(self, path, window_size, fee=0.001):
        self.path = path
        self.show_trade = show_trade
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.file_list = []
        self.load_from_csv()

        #features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.window_shape = (self.window_size, self.n_features+4)

        #action
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        new_df = pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(new_df)
        self.df = extractor.add_bar_features()

        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 
            'close']
        self.df.dropna(inplace=True)
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values

    def render_state(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def env(self, action, position):
        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        self.action = HOLD
        if action == BUY:
            if self.position == FLAT:
                self.position = LONG
                self.action = BUY
                self.entry_price = self.closingPrice
            elif self.position == SHORT:
                self.position = FLAT
                self.position = BUY
                self.exit_price = self.closingPrice
                self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1
                self.show_balance = self.show_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_short += 1
            elif action == 1:
                if self.position == FLAT:
                    self.position = SHORT
                    self.action = 1
                    self.entry_price = self.closingPrice
                elif self.position == LONG:
                    self.position = FLAT
                    self.action = 1
                    self.exit_price = self.closingPrice
                    self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                    self.show_balance = self.show_balance * (1.0 + self.reward)
                    self.entry_price = 0
                    self.n_long += 1

            if (self.position == LONG):
                temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                new_account = self.balance * (1.0 + temp_reward)
            elif(self.position == SHORT):
                temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
                new_account = self.show_balance * (1.0 + temp_reward)
            else:
                temp_reward = 0
                new_account = self.show_balance

            self.account = new_account
            self.current_tick += 1
            if(self.show_trade and self.current_tick%100 == 0):
                print("Tick: {0}/ Account (USD): {1}".format(self.current_tick, self.account))
                print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
            self.history.append((self.action, self.current_tick, self.closingPrice, self.account, self.reward))
            self.updateState()
            if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
                self.finish = True
                self.reward = self.get_profit()
            return self.state, self.reward, self.finish, {'account': np.array([self.account]),
                                                          'history': self.history,
                                                          'n_trades':{'long':self.n_long, 'short':self.n_short}}

    def get_profit(self):
        if(self.position == LONG):
            profit = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            profit = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit 
    
    #reset variables to initial
    def reset(self):
        self.current_tick = 0
        print("start episode ... {0} at {1}".format(self.rand_episode, self.current_tick))

        self.n_long = 0
        self.n_short = 0

        self.history = [] # show history 
        self.show_balance = 10000 #inital balance
        self.account = float(self.show_balance) #show balance
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False 

        self.updateState()
        return self.state

    def updateState(self):
        def one_hot_encoding(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrice[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encoding(prev_position, 3)
        profit = self.get_profit()
        self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
        return self.state