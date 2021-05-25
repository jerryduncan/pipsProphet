import numpy as np 
import tensorflow as tf 

#import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM 
from keras.optimizers import Adam

#import keras-rl agent 
from rl.agents.dqn import DQNAgent 
from rl.policy import BoltzmannPolicy, EpsGreedyQPolicy 
from rl.memory import SequentialMemory 

#working with trader environment
from TraderView import OhlcvEnv
# custom normalizer 
from util import Normalizerprocessor 

#ultulizing trader environment
def create_model(shape, nb_actions):
    model = Sequential()
    model.add(LSTM(64, input_shape=shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

def main():
    ENV_NAME = 'OHLCV-v0'
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

    # finally, we configure and compile our agent
    memory = SequentialMemory(limit=50000, window_length=TIME_STEP)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    # enable the dueling network 
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=Normalizerprocessor())
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    while Train:
        #train
        dqn.load_weights('')
        dqn.fit(env, nb_steps=5500, nb_max_episode_steps=10000, visualize=True, verbose=2)
        #validate 
        info = dqn.test(env_test, nb_episodes=1, visualize=True)
        n_long, n_short, total_reward, account = info['n_trades']['long'], info['n_trades']['short'], info[
            'total_reward'], int(info['account'])
        np.array([info]).dump(
            './info/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.info'.format(ENV_NAME, account, n_long, n_short,
                                                                    total_reward))
        dqn.save_weights(
            './model/duel_LSTM_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.h5f'.format(ENV_NAME, account, n_long, n_short,
                                                                    total_reward), overite=True)
                                                                    
if __name__ == '__main__':
    main()