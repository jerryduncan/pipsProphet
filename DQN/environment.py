from __future__ import print_function
from __future__ import absolute_import 
from __future__ import unicode_literals
from __future__ import division 

#loading up the environments requirements
import matplotlib.pyplot as plt
import IPython
import base64
import pandas
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_agents
import pandas_datareader as data_reader

from tqdm import tqdm_notebook, tqdm
from collections import deque


#loading up tensrflow_agents dependencies 
from tf_agents.agents.dqn import dqn_agent 
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils 
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym 
from tf_agents.environments import time_step as ts 
from tf_agents.metrics import tf_metrics 
from tf_agents.networks import q_network 
from tf_agents.specs import array_spec


class Environment(object):

    def reset():
        pass

    def step():
        pass

class ForexEnv(Environment):
    def __init__(self, config):
        pass 

class ReinforceEnvironment(Environment):
    def __init__(self, config):
        pass



if __name__=='__main__':
    nsteps = 5
    np.random.seed(448)

    env = ForexEnv(mode = 'train')
    time, obs, price = env.reset()
    t = 0
    
    done = False 
    while not done:
        action = np.random.randint(3)
        time, obs, price, done = env.setp(action)
        t += 1
        print(time)
        print(obs.shape)
        print(price)
        done = done or t==nsteps