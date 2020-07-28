import random
import sys
from argparse import ArgumentParser
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import DQNModel

Experience = namedtuple('Experience', 'state action reward next_state done')


class ReplayBuffer:
    def __init__(self, size: int = 1000):
        self.size = size
        self.buffer = []
        self.cursor = 0

    def sample_batch(self, n):
        return random.sample(self.buffer, n)

    def add(self, item: Experience):
        if len(self.buffer) < self.size:  # Still have room
            self.buffer.append(item)
        else:  # Overwrite item at cursor and move cursor
            self.buffer[self.cursor] = item
            self.cursor = (self.cursor + 1) % self.size


class Agent:
    def __init__(self,
                 net: DQN,
                 eps=0.05,
                 seed=0,
                 game='Breakout-v0',
                 buffer_size: int = 1000):
        self.env = gym.make(game)
        self.env.seed(seed)
        self.reset()

        self.eps = eps
        self.net = net
        self.buf = ReplayBuffer(buffer_size)
        self.state = self.env.reset()

    def reset(self):
        self.state = self.env.reset()

    def step(self, state: np.ndarray):
        # Choose action
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            q_values = self.net(torch.tensor(state).to(self.net.device))
            action = q_values.argmax(dim=1)

        # Take step, observing reward
        next_state, reward, done = self.env.step(action)

        # Store into replay buffer
        self.buf.add(Experience(self.state, action, reward, next_state, done))

        # Advance to next state
        self.state = next_state

        if done:
            self.reset()

        return reward, done


class DQN(LightningModule):
    def __init__(self, hparms):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {
            'test_loss': test_loss,
            'n_correct_pred': n_correct_pred,
            'n_pred': len(x)
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader()

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--game',
                            '-g',
                            default='Breakout-v0',
                            choices=['Breakout-v0', 'Pong-v0'])
        return parser


def run(s=[]):
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = DQN.add_model_specific_args(parser)
    hparams = parser.parse_args(s)

    checkpoint_callback = ModelCheckpoint(verbose=True,
                                          monitor='avg_q',
                                          save_top_k=1,
                                          save_weights_only=False,
                                          mode='max')

    model = DQN(hparams)
    print(model)
    trainer = Trainer.from_argparse_args(
        hparams, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run(sys.argv[1:])
