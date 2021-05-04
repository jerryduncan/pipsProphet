# PipsProphet -- Predicting price action with Deep Reinforcement Learning

The forex and synthetic indices market has witnessed huge changes in prices from high, low, low-high and high-low due to fundamental and technical analysis which intends to determine the actual state of price at each point in time-frame, this has led many financial experts in the market to look beyond the market to proffer financial advice to traders across the world.

Alot will ague that the use of INDICATORS such as [Fibonacci Retracement](https://en.wikipedia.org/wiki/Fibonacci_retracement), [Moving Average](https://en.wikipedia.org/wiki/Moving_average), [Stochastic Oscillator](https://en.wikipedia.org/wiki/Stochastic_oscillator), [Relative Strength Index](https://en.wikipedia.org/wiki/Relative_strength_index), [Bollinger Bands](https://origin2.cdn.componentsource.com/sites/default/files/resources/dundas/538216/Documentation/Bollinger.html) and alot more which basically are built on mathematical functions are better suit in understanding price, another set of financial experts prefer to study price chart over a long period of time using the Higher TimeFrmae HTF and Lower TimeFrame LTF index which they call PRICE ACTION.

This algorithm intends to merge both the indicators and price action understanding with deep learning algorithms to understudy price and help forecast price over a short and long period of time.

# Model Architecture
Our architecture involves building a feedforward backprogrational neural network acting as a Q-network. This network consists of 3 hidden layers of 20 ReLU neurons each, followed by an output layer of 3 linear neurons. Both hidden layers will be simulated inside the financial market.

The state of the hidden layers was set experimentally, while the three linear output neurons are inherent to our system design: each representing the Q-value of a give action.

Our network interacts within a simulated market environment in discrete  steps t = 0,1,2,... receiving a state vector S<sub>t</sub> as input at each of those steps. After a forward propagation, each of the three linear neurons outputs the Q-network current estimate for an action value <i>Q<sub>an</sub> (S<sub>t</sub>, W<sub>k</sub>)</i> for each of the three possible outcomes n <i>E</i> [0,1,2], where W<sub>k</sub> represent the set of network weights after k updates.

The estimates <i>Q<sub>an</sub> (S<sub>t</sub>, W<sub>k</sub>)</i> are fed to a e-greedy action selection method which selects the action choice for step t as either <i>A<sub>t</sub> = arg max<sub>a</sub></i> <i>Q<sub>a</sub> (S<sub>t</sub>, W<sub>k</sub>).</i>

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

<<<<<<< HEAD
# Market simulation
We created an environment to coordinate the flow of information that reaches the system so that it follows the reinforcement learning paradigm, supplying the system with a state, receiving its response in the form of an action and answering with a new state and reward. Our environment is consistent with the real foreign exchange market, so that its learned behaviour and our measure of its performance would translate to real trading.

The market simulation follows prices from a tick dataset T = {T0, .., TD}. The system is only allowed to make a decision every time <i>skip_ticks</i> ticks. At each step/interaction <i>t</i> the market environment is at the price in tick T<sub>i=t·time_skip+b,</sub> where <i>b</i> is the chosen starting tick for the first interaction, and sends the system the state S<sub>t</sub>

A response is received in the form of an action signal A<sub>t</sub>, after which the market environment skips to the price in T<sub>i+time_skip</sub> and drafts a new state S<sub>t+1</sub> and a scalar reward R<sub>t+1</sub> for the action A<sub>t</sub>, which are sent to the system

These interactions continue until the end of dataset is reached, completing the pass through the dataset.

# Training Process
The training process is structured into epochs. Each epoch has four phases, a structured learning phase and three phases to assess the learning progress:

- Training pass over a training dataset (learning phase)
- Evaluation of Q-values over random set of states (first metric)
- Test over the training dataset (second metric)
- Test over the validation dataset (third metric)

For the first metric, states are collected by running a random policy through the training dataset and then at each epoch we evaluate the Q-network's average estimated Q-value for that set of states. A smooth growth in this metric, with no divergence validates that the Q-Network is learning and stable.

For the second and third metrics the profit generated over the test is recorded and the evolution of that profit over the epocsh is the indicator of how well the system is learning

# Learning Function 
The role of the learning function is to receive the transitions <i>e<sub>t</sub></i> observed during learning passes and use them to change the Q-Network’s weights in a way that improves its approximation of <i>q</i>

=======
![image](https://user-images.githubusercontent.com/41350149/117012921-51849b80-ace7-11eb-93c2-6a0b608a0f9e.png)
>>>>>>> 9a2fcc4153b8a18b3dc32e4fad5c907c7c280e41


# Contributing 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
