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

