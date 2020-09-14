import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from random import sample
from typing import Tuple

import gym
import numpy as np
import torch
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from model import DQN
from utils import float01, huber, toInt


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

    def append(self, x: Tuple[torch.Tensor, int, float, torch.Tensor, bool]):
        self.buffer[self.index] = x
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        experiences = [self.buffer[index] for index in indices]

        states, actions, rewards, next_states, dones = zip(*experiences)

        # stack will copy the tensors, so .to() will not affect
        # the data in the buffer
        states = torch.stack(states, dim=0).to(self.device) / 255.0
        next_states = torch.stack(next_states, dim=0).to(self.device) / 255.0

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards,
                               dtype=torch.float32,
                               device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self,
                 game: str,
                 replay_buffer_capacity: int,
                 replay_start_size: int,
                 batch_size: int,
                 discount_factor: float,
                 lr: float,
                 device: str = 'cuda:0',
                 env_seed: int = 0,
                 frame_buffer_size: int = 4,
                 print_self=True):

        self.device = device
        self.discount_factor = discount_factor
        self.game = game
        self.batch_size = batch_size

        self.replay_buf = ReplayBuffer(capacity=replay_buffer_capacity)

        self.env = FrameStack(
            AtariPreprocessing(
                gym.make(self.game),
                # noop_max=0,
                # terminal_on_life_loss=True,
                scale_obs=False),
            num_stack=frame_buffer_size)
        self.env.seed(env_seed)
        self.reset()

        self.n_action = self.env.action_space.n
        self.policy_net = DQN(self.n_action).to(self.device)
        self.target_net = DQN(self.n_action).to(self.device).eval()
        self.optimizer = RMSprop(
            self.policy_net.parameters(),
            alpha=0.95,
            # momentum=0.95,
            eps=0.01)

        if print_self:
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
        self.state = self.env.reset()

    @torch.no_grad()
    def step(self, epsilon, clip_reward=True):
        """
        Choose an action based on current state and epsilon-greedy policy
        """
        # Choose action
        if random.random() <= epsilon:
            q_values = None
            action = self.env.action_space.sample()
        else:
            torch_state = torch.tensor(self.state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0) / 255.0
            q_values = self.policy_net(torch_state)
            action = int(q_values.argmax(dim=1).item())

        # Apply action
        next_state, reward, done, _ = self.env.step(action)
        if clip_reward:
            reward = max(-1.0, min(reward, 1.0))

        # Store into replay buffer
        self.replay_buf.append(
            (torch.tensor(
                np.array(self.state), dtype=torch.float32, device="cpu") / 255.,
             action, reward,
             torch.tensor(
                 np.array(next_state), dtype=torch.float32, device="cpu") / 255.,
             done))

        # Advance to next state
        self.state = next_state
        if done:
            self.reset()

        return reward, q_values, done

    def q_update(self):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = [
            x.to(self.device) for x in self.replay_buf.sample(self.batch_size)
        ]

        with torch.no_grad():
            y = torch.where(
                dones, rewards, rewards +
                self.discount_factor * self.target_net(next_states).max(1)[0])

        predicted_values = self.policy_net(states).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        loss = huber(y, predicted_values, 2.)
        loss.backward()
        self.optimizer.step()
        return (y - predicted_values).abs().mean()


def save(agent, path):
    os.makedirs(path, exist_ok=True)
    ckpt = os.path.join(path, f'{agent.game}.pt')
    torch.save(agent.policy_net.state_dict(), ckpt)


def parse_args(argv):
    p = ArgumentParser()
    p.add_argument('--game', default='BreakoutNoFrameskip-v0')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--frames', type=toInt, default=int(5e6))
    p.add_argument('--max_eps', default=1.0, type=float01)
    p.add_argument('--min_eps', default=0.1, type=float01)
    p.add_argument('--eps_duration', type=toInt, default=int(1e6))
    p.add_argument('--replay_buffer_capacity', type=toInt, default=int(1e5))
    p.add_argument('--replay_start_size',
                   type=toInt,
                   default=int(1e5),
                   help='init random steps')
    p.add_argument('--discount_factor', default=0.99, type=float01)
    p.add_argument('--target_update_frequency',
                   default=int(1e4),
                   help='update target net every N iterations')
    p.add_argument('--frame_skip',
                   default=4,
                   help='number of frames between agent decisions')
    p.add_argument('--policy_update_frequency',
                   type=int,
                   default=4,
                   help='number of frames between gradient steps')
    p.add_argument('--episode_length', type=int, default=1000)
    p.add_argument('--lr', type=float, default=0.00025)
    p.add_argument('--save_path', type=str, default='checkpoints')
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
        lr=args.lr,
    )
    time = datetime.now().isoformat()
    writer = SummaryWriter(f'tensorboard_logs/{time}')

    torch.manual_seed(0)
    np.random.seed(0)

    # Train agent
    print('Begin training...')
    episode_reward = 0.
    episode_steps = 0
    try:
        for i in trange(args.frames, desc='Frames', leave=True):
            # Compute epsilon
            epsilon = max(
                args.min_eps,
                1 - i * (args.max_eps - args.min_eps) / args.eps_duration)

            if i % 1000 == 0:
                writer.add_scalar('Epsilon', epsilon, i)

            if i % 10000 == 0:
                for name, p in agent.policy_net.named_parameters():
                    writer.add_histogram('policy' + name, p, bins='auto')

                for name, p in agent.target_net.named_parameters():
                    writer.add_histogram('target' + name, p, bins='auto')

            # Act on next frame
            reward, q_values, done = agent.step(epsilon)

            if q_values is not None:
                q_values = q_values.squeeze()
                for j in range(agent.n_action):
                    writer.add_scalar(f'Q_values/{j}', q_values[j], i)

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
                agent.target_net.eval()

            if done:
                writer.add_scalar('Reward_per_step',
                                  episode_reward / episode_steps, i)
                writer.add_scalar('Total_episode_reward', episode_reward, i)
                writer.add_scalar('Episode_length', episode_steps, i)
                episode_reward = 0.
                episode_steps = 0

    except KeyboardInterrupt:
        pass
    if args.save_path is not None:
        save(agent, args.save_path)


if __name__ == '__main__':
    run(sys.argv[1:])
