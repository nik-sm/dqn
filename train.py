import random
import sys
from argparse import ArgumentParser
from collections import deque, namedtuple
from random import sample

import numpy as np

import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ftransforms
from model import DQN
from torch.optim import Adam
from tqdm import trange
from utils import float01

Experience = namedtuple('Experience', 'state action reward next_state done')


class ReplayBuffer:  # (IterableDataset):
    """Store N previous experiences for random sampling"""
    def __init__(self, capacity, batch_size, device='cuda:0'):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0
        self.device = device

    def __repr__(self):
        return (f'ReplayBuffer: capacity {self.capacity}, ' +
                f'size {self.size}, batch_size {self.batch_size}')

    def append(self, x):
        self.buffer[self.index] = x
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def sample(self):
        indices = sample(range(self.size), self.batch_size)
        experiences = [self.buffer[index] for index in indices]
        return [
            torch.stack(x, dim=0).to(self.device) for x in zip(*experiences)
        ]


class Agent:
    def __init__(self,
                 game: str,
                 replay_buffer_size: int,
                 replay_start_size: int,
                 batch_size: int,
                 discount_factor: float,
                 device: str = 'cuda:0',
                 env_seed: int = 0,
                 frame_buffer_size: int = 4):

        self.device = device
        self.discount_factor = discount_factor
        self.game = game

        self.frame_buf = deque(maxlen=frame_buffer_size)
        self.replay_buf = ReplayBuffer(replay_buffer_size, batch_size)

        self.env = gym.make(self.game)
        self.env.seed(env_seed)
        self.reset()

        n_action = self.env.action_space.n
        self.policy_net = DQN(n_action).to(self.device)
        self.target_net = DQN(n_action).to(self.device)
        self.optimizer = Adam(self.policy_net.parameters())

        print(self)
        self._fill_replay_buf(replay_start_size)

    def __repr__(self):
        return '\n'.join([
            'Agent:',
            f'Game: {self.game}',
            f'Device: {self.device}',
            f'Policy net: {self.policy_net}',
            f'Target net: {self.target_net}',
            f'Replay buf: {self.replay_buf}',
            f'Frame buf size: {self.frame_buf.maxlen}',
        ])

    def _fill_replay_buf(self, replay_start_size):
        for _ in trange(replay_start_size,
                        desc='Fill replay_buf randomly',
                        leave=True):
            self.step(1.0)

    def process_frame(self, x):
        """preprocess frame along with the 3 previously seen frames"""
        x = ftransforms.to_pil_image(x)
        x = ftransforms.to_grayscale(x)

        x = ftransforms.resize(x, [110, 84])
        x = ftransforms.center_crop(x, 84)
        x = ftransforms.to_tensor(x)

        # Handle flicker - TODO unnecessary with Openai gym?
        # NOTE - need to add to frame buf before modifying,
        # otherwise all the max values get retained forever
        # try:
        #     x = torch.max(x, self.frame_buf[-1])
        # except IndexError:
        #     pass

        # right-append frame, dropping frame 0
        self.frame_buf.append(x)

    def state_from_frame_buf(self):
        return torch.cat(tuple(self.frame_buf), dim=0).to(self.device)

    def reset(self):
        """Reset the end, pre-populate self.frame_buf and self.state"""
        x = self.env.reset()
        for _ in range(self.frame_buf.maxlen):
            self.process_frame(x)
        self.state = self.state_from_frame_buf()

    @torch.no_grad()
    def step(self, epsilon):
        """
        Choose an action based on current state and epsilon-greedy policy
        """
        # Choose action
        if random.random() <= epsilon:
            action = torch.tensor([self.env.action_space.sample()
                                   ]).to(self.device)
        else:
            q_values = self.policy_net(self.state.unsqueeze(0))
            action = q_values.argmax(dim=1)

        # Take step, observing reward
        frame, reward, done, _ = self.env.step(action)

        # Process frame and obtain 4-frame state
        self.process_frame(frame)
        next_state = self.state_from_frame_buf()

        # Store into replay buffer
        self.replay_buf.append(
            Experience(self.state, torch.tensor(action), torch.tensor(reward),
                       next_state, torch.tensor(done)))

        # Advance to next state
        self.state = next_state.to(self.device)
        if done:
            self.reset()

        return reward, done

    def q_update(self):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = self.replay_buf.sample()
        y = torch.where(
            dones, rewards, rewards + self.discount_factor *
            torch.max(self.target_net(next_states), dim=1)[0])
        predicted_values = torch.gather(self.policy_net(states), 1,
                                        actions).squeeze()
        loss = F.mse_loss(y, predicted_values)
        loss.backward()
        self.optimizer.step()


def parse_args(argv):
    p = ArgumentParser()
    p.add_argument('--game',
                   default='Breakout-v0',
                   choices=['Breakout-v0', 'Pong-v0'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--initial_exploration', default=1.0, type=float01)
    p.add_argument('--final_exploration', default=0.1, type=float01)
    p.add_argument('--final_exploration_frame', default=int(1e6), type=int)
    p.add_argument('--replay_buffer_size', type=int, default=int(1e6))
    p.add_argument('--replay_start_size',
                   type=int,
                   default=int(5e4),
                   help='initial random steps to populate buffer')
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
    p.add_argument('--policy_update_frequency',
                   type=int,
                   default=4,
                   help='number of frames between gradient steps')
    p.add_argument('--episode_length', type=int, default=1000)
    return p.parse_args(argv)


def run(argv=[]):
    args = parse_args(argv)

    # Setup
    print('Create agent...')
    agent = Agent(
        game=args.game,
        replay_buffer_size=args.replay_buffer_size,
        replay_start_size=args.replay_start_size,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    # Train agent
    print('Begin training...')
    for e in trange(args.episodes, desc='Episodes', leave=True):
        for i in trange(args.episode_length, desc='Frames', leave=False):
            # Compute epsilon
            epsilon = float(args.final_exploration - args.initial_exploration)
            epsilon /= args.final_exploration_frame
            epsilon *= e * args.episode_length + i

            # Act on next frame
            agent.step(epsilon)

            # Every K steps, update policy net
            if i % args.policy_update_frequency == 0:
                agent.q_update()

            # Every C steps, update target net
            if i % args.target_update_frequency == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())


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

if __name__ == '__main__':
    run(sys.argv[1:])
