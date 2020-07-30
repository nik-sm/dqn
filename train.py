import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from random import sample

import numpy as np

import gym
import torch
import torch.nn.functional as F
from gym.wrappers import AtariPreprocessing, FrameStack
from model import DQN
from torch.optim import Adam
# from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils import float01, toInt


class ReplayBuffer:
    """Shared buffer for all worker processes"""
    def __init__(self, capacity, device='cuda:0'):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.index = 0
        self.size = 0
        self.device = device

    def __repr__(self):
        return (f'ReplayBuffer: capacity {self.capacity}, ' +
                f'size {self.size}')

    def append(self, x):
        self.buffer[self.index] = x
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        experiences = [self.buffer[index] for index in indices]

        states, actions, rewards, next_states, dones = zip(*experiences)

        # stack will copy the tensors, so .to() will no affect the data in the buffer
        states = torch.stack(states, dim=0).to(self.device)
        next_states = torch.stack(next_states, dim=0).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, dones


# class ExperienceDataset(IterableDataset):
#     """Thin wrapper dataset for multiprocess loading"""
#     def __init__(self, buf, batch_size):
#         self.buf = buf
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         while True:
#             yield self.buf.sample(self.batch_size)


class Agent:
    def __init__(self,
                 game: str,
                 replay_buffer_capacity: int,
                 replay_start_size: int,
                 batch_size: int,
                 discount_factor: float,
                 device: str = 'cuda:0',
                 env_seed: int = 0,
                 frame_buffer_size: int = 4):

        self.device = device
        self.discount_factor = discount_factor
        self.game = game
        self.batch_size = batch_size

        self.replay_buf = ReplayBuffer(capacity=replay_buffer_capacity)
        # self.loader_it = iter(
        #     DataLoader(ExperienceDataset(self.replay_buf, batch_size),
        #                num_workers=0,
        #                pin_memory=True,
        #                batch_size=None))

        self.env = FrameStack(AtariPreprocessing(gym.make(self.game),
                                                 noop_max=0,
                                                 terminal_on_life_loss=True,
                                                 scale_obs=False),
                              num_stack=frame_buffer_size)
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
            'Agent:', f'Game: {self.game}', f'Device: {self.device}',
            f'Policy net: {self.policy_net}', f'Target net: {self.target_net}',
            f'Replay buf: {self.replay_buf}'
        ])

    def _fill_replay_buf(self, replay_start_size):
        for _ in trange(replay_start_size,
                        desc='Fill replay_buf randomly',
                        leave=True):
            self.step(1.0)

    def reset(self):
        """Reset the end, pre-populate self.frame_buf and self.state"""
        self.state = torch.tensor(self.env.reset(), dtype=torch.float32) / 255.

    @torch.no_grad()
    def step(self, epsilon):
        """
        Choose an action based on current state and epsilon-greedy policy
        """
        # Choose action
        if random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            torch_state = torch.tensor(self.state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0) / 255.0
            q_values = self.policy_net(torch_state)
            action = int(q_values.argmax(dim=1).item())

        # Apply action
        next_state, reward, done, _ = self.env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32) / 255.
        if reward > 0:
            reward = 1.0
        elif reward < 0:
            reward = -1.0

        # Store into replay buffer
        self.replay_buf.append(
            (torch.tensor(np.array(self.state), dtype=torch.float32, device="cpu"), action, reward,
             torch.tensor(np.array(next_state), dtype=torch.float32, device="cpu"), done))

        # Advance to next state
        self.state = next_state
        if done:
            self.reset()

        return reward, done

    def q_update(self):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = [
            x.to(self.device) for x in self.replay_buf.sample(self.batch_size)
        ]

        y = torch.where(
            dones, rewards, rewards + self.discount_factor *
            torch.max(self.target_net(next_states), dim=1)[0])
        predicted_values = torch.gather(self.policy_net(states), 1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(y, predicted_values)
        loss.backward()
        self.optimizer.step()
        return (y - predicted_values).abs().mean()


def parse_args(argv):
    p = ArgumentParser()
    p.add_argument('--game', default='BreakoutNoFrameskip-v0')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--frames', type=toInt, default=int(5e6))
    p.add_argument('--initial_exploration', default=1.0, type=float01)
    p.add_argument('--final_exploration', default=0.1, type=float01)
    p.add_argument('--final_exploration_frame', type=toInt, default=int(1e6))
    p.add_argument('--replay_buffer_capacity', type=toInt, default=int(1e6))
    p.add_argument('--replay_start_size',
                   type=toInt,
                   default=5e4,
                   help='init random steps')
    p.add_argument('--discount_factor', default=0.99, type=float01)
    p.add_argument('--target_update_frequency',
                   default=1e4,
                   help='update target net every N iterations')
    p.add_argument('--frame_skip',
                   default=4,
                   help='number of frames between agent decisions')
    p.add_argument('--policy_update_frequency',
                   type=int,
                   default=4,
                   help='number of frames between gradient steps')
    p.add_argument('--episode_length', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args(argv)


def run(argv=[]):
    args = parse_args(argv)

    # Setup
    print('Create agent...')
    agent = Agent(
        game=args.game,
        replay_buffer_capacity=args.replay_buffer_capacity,
        replay_start_size=args.replay_start_size,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
    )
    time = datetime.now().isoformat()
    writer = SummaryWriter(f'tensorboard_logs/{time}')

    torch.manual_seed(0)
    np.random.seed(0)

    # Train agent
    print('Begin training...')
    episode_reward = 0.
    episode_steps = 0
    bar = trange(args.frames, desc='Frames', leave=True)
    for i in bar:
        # Compute epsilon
        epsilon = float(abs(args.final_exploration - args.initial_exploration))
        epsilon /= args.final_exploration_frame
        epsilon = max(args.final_exploration, 1 - i * epsilon)
        if i % 1000 == 0:
            bar.set_postfix(epsilon=epsilon)

        # Act on next frame
        reward, done = agent.step(epsilon)
        episode_reward += reward
        episode_steps += 1
        writer.add_scalar('Reward', reward, i)

        # Every K steps, update policy net
        if i % args.policy_update_frequency == 0:
            td_error = agent.q_update()
            writer.add_scalar('TD_Error', td_error, i)

        # Every C steps, update target net
        if i % args.target_update_frequency == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if done:
            writer.add_scalar('Avg_Episode_Reward',
                              episode_reward / episode_steps, i)
            agent.reset()
            episode_reward = 0.
            episode_steps = 0


if __name__ == '__main__':
    run(sys.argv[1:])
