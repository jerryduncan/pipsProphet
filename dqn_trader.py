from __future__ import absolute_import 
from __future__ import unicode_literals 
from __future__ import print_function
from __future__ import division 

import numpy as np 

#import keras 
import tensorflow as tensorflow
from tensorflow import keras

#import keras-rl agent 
from rl.agents.dqn import DQNAgent 
from rl.policy import BoltzmannPolicy, EpsGreedyQPolicy 
from rl.memory import SequentialMemory 

#working with trader environment
from TraderView import OhlcvEnv
from util import Normalizerprocessor 

#ultulizing trader environment



if __name__ == '__main__':
    main()