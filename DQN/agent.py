from __future__ import print_function
from __future__ import absolute_import 
from __future__ import unicode_literals 
from __future__ import division

#import dependencies 
import numpy as np 
import math
import tensorflow as tf 
import tf_agents 



#define agent state 
class PipsProphetAgent(object):
    def __init__(self, set_action, reward_function):
        self.set_action = set_action
        self.reward_function = reward_function
        self.total_reward = 0
    
    def reset_total_reward(self):
        self.total_reward = 0

    def update_buffer(self, observation_history, action_history):
        pass

    def learn_from_buffer(self):
        pass

    def action(self, observation_history, action_history):
        pass

    def __str__(self):
        pass

    