# H1 Overview

DQN works by using a Neural Network as a function approximator. Where the function is trying to determine the action which gives the highest reward
given the current observed state.

# H1 Keywords
**Episode**: a single simulation of the environment
**Tensor**: a n-dimension array/matrix which can be used for GPU processing

# H1 Replay Memory

Replay memory is a deque (***double-ended queue***) used to train the DQN, it stores the (state, action) transitions from state x to state y.
It is a cyclic buffer with a bounded size that holds the transitions observed recently. It also implements
sample method for selecting a random batch of transitions for training.

# H1 Training

For a given number of episodes repeat
1. select action (either random or policy depending on the threshold)
2. step the environment
3. store the transition in the memory
4. update current state
5. optimize the model
6. soft update the target network's weights

# Optimization
For each step of optimization the procedure follows:
1. sample from memory of transitions a given BATCH_SIZE
2. create a mask of non-final-next-states & get the non final next states
3. split batch into 3 lists: state_batch, action_batch and reward_batch
4. run the state_batch through the NN and select the columns of actions taken.
5. 
