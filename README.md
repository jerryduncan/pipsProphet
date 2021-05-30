# PipsProphet -- Predicting price action with Deep Reinforcement Learning

The forex and synthetic indices market has witnessed huge changes in prices from high, low, low-high and high-low due to fundamental and technical analysis which intends to determine the actual state of price at each point in time-frame, this has led many financial experts in the market to look beyond the market to proffer financial advice to traders across the world.

Alot will ague that the use of INDICATORS such as [Fibonacci Retracement](https://en.wikipedia.org/wiki/Fibonacci_retracement), [Moving Average](https://en.wikipedia.org/wiki/Moving_average), [Stochastic Oscillator](https://en.wikipedia.org/wiki/Stochastic_oscillator), [Relative Strength Index](https://en.wikipedia.org/wiki/Relative_strength_index), [Bollinger Bands](https://origin2.cdn.componentsource.com/sites/default/files/resources/dundas/538216/Documentation/Bollinger.html) and alot more which basically are built on mathematical functions are better suit in understanding price, another set of financial experts prefer to study price chart over a long period of time using the Higher TimeFrmae HTF and Lower TimeFrame LTF index which they call PRICE ACTION.

This algorithm intends to merge both the indicators and price action understanding with deep learning algorithms to understudy price and help forecast price over a short and long period of time.

# Model Architecture
Our architecture involves building a feedforward backprogrational neural network acting as a Q-network. This network consists of 3 hidden layers of 20 ReLU neurons each, followed by an output layer of 3 linear neurons. Both hidden layers will be simulated inside the financial market.

The state of the hidden layers was set experimentally, while the three linear output neurons are inherent to our system design: each representing the Q-value of a give action.

Our network interacts within a simulated market environment in discrete  steps t = 0,1,2,... receiving a state vector S<sub>t</sub> as input at each of those steps. After a forward propagation, each of the three linear neurons outputs the Q-network current estimate for an action value ![equation](https://latex.codecogs.com/gif.latex?Q_%7Ban%7D%28S_%7Bt%7D%2CW_%7Bk%7D%29) for each of the three possible outcomes ![equation](https://latex.codecogs.com/gif.latex?n%20%5Cvarepsilon%20%5B0%2C1%2C2%5D), where W<sub>k</sub> represent the set of network weights after k updates.

The estimates ![equation](https://latex.codecogs.com/gif.latex?Q_%7Ban%7D%28S_%7Bt%7D%2CW_%7Bk%7D%29) are fed to a e-greedy action selection method which selects the action choice for step t as either ![equation](https://latex.codecogs.com/gif.latex?A_%7Bt%7D%20%3D%20argmax_%7Ba%7DQ_%7Ban%7D%28S_%7Bt%7D%2CW_%7Bk%7D%29)

There is an external influence on the agent to invest <i>position_size</i> of the choosen asset at a time, a value set by the user, leaving it with five actions: open long position, open short position, close long position, close short position and do nothing.

The selected action At is then received by the simulated market environment. With role to provide a acurate simulation of the foreign exchange market and coordinate the flow of information that reaches the system so that it follows the reinforcement learning paradigm

Each state simulated by the environment conditions includes the following information

- Type of current open position;
- Value of any open position in view of simulated market current prices Bid<sub>i</sub> and Ask<sub>i</sub>, where <i>i</i> is an index over the entries in the dataset used by the market ;
- Current size of trading account;
- Feature vector F<sub>i</sub>; (created using the market data entries, by the preprocessing stage inspired by the technical analysis approach)

As for the reward given to the network backpropagation, each action is rewarded as follows:

- Opening a position is rewarded by the unrealized profit it creates;
- keeping a position open is rewarded by the fluctuation of the position unrealized profit;
- Closing a position is rewarded with the attained profit 
- Doing nothing receives zero reward 

![image](https://user-images.githubusercontent.com/41350149/117012921-51849b80-ace7-11eb-93c2-6a0b608a0f9e.png)

# Intialize Environment 
```
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
```

# Reference
(Guide - RL) Reinforcement Q-Learning from Scratch in Python with OpenAI Gym
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

Train a Deep Q Network with TF-Agents https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

Deep Reinforcement Learning for Automated Stock Trading https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

Trading Environment(OpenAI Gym) + DDQN (Keras-RL) https://github.com/miroblog/deep_rl_trader

# Pull requests 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
