
"""
### Reinforcement Learning with [DQN](https://en.wikipedia.org/wiki/Q-learning#Variants)

Regarding v5:

 This world is where we will build scaffolding for each world to get DQNs set up. As of now the world is where we have deep q reinforcement learning happening for a single buyer agent...

**Deep** BananaWorld:

- 1 x 10 grid
- Buyer starts at (1,4), Seller is at (1,10)

- seller makes bananas every turn (expenses are 0)
- seller has very high inventory (1000)

- bananas have very high shelf life (1000)
- bananas cost 1

- buyer gets income of .5
- buyer has very high inventory (1000)
- buyer has starting salary of 5
- buyer has health drop rate of 0 (live forever)
- buyer has strategy to greedily buy the cheapest banana (only one banana)

Goal:

buyers have 1 action they can do at each timestep, either move 1 position OR buy bananas...
therefore the purpose of this world is for the buyer to move closer to the seller to minimize price cost

buyers get 1 reward only if their wealth is 2.5 or greater, if it is 0 they will not get a reward

state space is as follows: self.pos, buyer pos
reward = 2^(-1 * distance from buyer)
action space is as follows: move up, move down

Below is the code which scaffolds the DeepQLearningNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) # TODO WILL NEED TO CHANGE to pass on  INIT!
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done=False):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
