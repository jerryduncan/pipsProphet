import numpy as np 

#import keras 
import tensorflow as tensorflow
from tensorflow import keras

#import keras-rl agent 
from rl.agents.dqn import DQNAgent 
from rl.policy import BoltzmannPolicy, EpsGreedyQPolicy 
from rl.memory import SequentialMemory 

#working with trader environment
from TraderEnv import OhlcvEnv
from util import Normalizerprocessor 

#build model 


if __name__ == '__main__':
    main()