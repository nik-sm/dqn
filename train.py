import random
import sys
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass
from random import sample
from typing import NamedTuple

import gym
import torch
import torchvision.transforms.functional as ftransforms
from model import DQN
from torch.utils.data import DataLoader, IterableDataset
from utils import float01


class Experience(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer(IterableDataset):
    """Store N previous experiences for random sampling"""
    def __init__(self, capacity, batch_size):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0

    def append(self, x):
        self.buffer[self.index] = x
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def sample(self):
        indices = sample(range(self.size), self.batch_size)
        return [self.buffer[index] for index in indices]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError('TODO - multiple workers')
        else:
            yield self.sample()


"""
Agent TODO:
- At time step t, set every pixel i = max(p_{i, t}, p_{i, t-1})
  (compensates for flickering, maybe handled by openai gym?)
- clamp rewards (-1, 0, 1)
- select action 1 of every k frames, repeat last action otherwise
- collect groups of 4 frames, preprocess
- "clip the error term from the update
r + gamma*max_a' Q_target(s', a') - Q_policy(s, a) to be between -1 and 1"
"""


@dataclass
class Agent:
    policy_net: DQN
    target_net: DQN
    loader: DataLoader
    optimizer: torch.optim.Optimizer
    game: str
    env_seed: int = 0
    replay_buffer_size: int = 1000
    frame_buffer_size: int = 4

    def __post_init__(self):
        self.env = gym.make(self.game)
        self.env.seed(self.env_seed)
        self.reset()

        self.replay_buf = ReplayBuffer(self.buffer_size)
        self.frame_buf = deque(maxlen=self.frame_buffer_size)

    def process_frame(self, x):
        """preprocess frame along with the 3 previously seen frames"""
        x = ftransforms.to_grayscale(x)
        x = ftransforms.to_tensor(x)
        # Handle flicker - TODO unnecessary with Openai gym?
        x = torch.max(x, self.frame_buf[-1])
        x = ftransforms.resize(x, [110, 84])
        x = ftransforms.center_crop(x, 84)
        self.frame_buf.append(x)

    def get_state(self):
        return torch.stack(tuple(self.frame_buf), dim=0)

    def reset(self):
        x = self.env.reset()
        for _ in range(self.frame_buffer_size):
            self.process_frame(x)
        self.state = self.get_state()

    def step(self, state: torch.Tensor, epsilon):
        # Choose action
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = state.to(self.policy_net.device)
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1)

        # Take step, observing reward
        frame, reward, done = self.env.step(action)

        # Process frame and obtain 4-frame state
        next_state = self.process_frame(frame)

        # Store into replay buffer
        self.buf.add(Experience(self.state, action, reward, next_state, done))

        # Advance to next state
        self.state = next_state
        if done:
            self.reset()

        return reward, done


def run(argv=[]):
    p = ArgumentParser()
    p.add_argument('--game',
                   default='Breakout-v0',
                   choices=['Breakout-v0', 'Pong-v0'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--initial_exploration', default=1.0, type=float01)
    p.add_argument('--final_exploration', default=0.1, type=float01)
    p.add_argument('--final_exploration_frame', default=int(1e6), type=int)
    p.add_argument('--replay_start_size', type=int, default=int(5e4))
    p.add_argument('--discount_factor', default=0.99, type=float01)
    p.add_argument('--target_update_frequency',
                   default=int(1e4),
                   help='update target net every N iterations')
    p.add_argument('--agent_history_length',
                   default=4,
                   help='number of consecutive frames used in network')
    p.add_argument('--action_repeat',
                   type=int,
                   default=4,
                   help='agent sees only every k\'th frame')
    p.add_argument('--update_frequency',
                   type=int,
                   default=4,
                   help='number of actions between gradient steps')
    args = p.parse_args(argv)

    print('Setup networks, optimizers, agent, and environment...')
    policy_net = DQN()
    target_net = DQN()
    loader = DataLoader()
    agent = Agent(
        policy_net=policy_net,
        target_net=target_net,
    )
    agent.train()

    print('Initialize replay buffer with random policy...')
    raise

    print('Begin training...')
    for e in range(args.episodes):
        pass

    raise
    if batch_idx % self.target_update_frequency == 1:
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == '__main__':
    run(sys.argv[1:])
