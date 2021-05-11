from __future__ import print_function
from __future__ import absolute_import 
from __future__ import unicode_literals
from __future__ import division 

#loading up the environments requirements
import matplotlib.pyplot as plt
import IPython
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf

#loading up tensrflow_agents dependencies 
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils 
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym 
from tf_agents.environments import time_step as ts 
from tf_agents.specs import array_spec

