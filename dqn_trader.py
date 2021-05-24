from __future__ import absolute_import 
from __future__ import unicode_literals 
from __future__ import print_function
from __future__ import division 

import numpy as np 

#import keras 
import tensorflow as tf 
from tensorflow import keras

#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten, LSTM 
#from keras.optimizers import Adam

# tensorflow agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.eval import metric_utils 
from tf_agents.metrics import tf_metrics

#import keras-rl agent 
#from rl.agents.dqn import DQNAgent 
#from rl.policy import BoltzmannPolicy, EpsGreedyQPolicy 
#from rl.memory import SequentialMemory 

#working with trader environment
from TraderView import OhlcvEnv
from util import Normalizerprocessor 

#ultulizing trader environment
def model_create(shape, nb_actions):
    model = Sequential()
    return model

def main():
    ENV_NAME = 'OHLCV'
    TIME_STEP = 30

    TRAIN_PATH = ""
    TEST_PATH = ""
    env_train = OhlcvEnv(TIME_STEP, path=TRAIN_PATH)
    env_test = OhlcvEnv(TIME_STEP, path=TEST_PATH)

    np.random.seed(456)
    env.seed(562)

    nb_actions = env.action_space.n
    model = model_create(shape=env.shape, nb_actions=nb_actions)
    print(model.summary())

if __name__ == '__main__':
    main()