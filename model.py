import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_action=4, input_shape=(4, 84, 84)):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        shape_after = shape_after_conv(input_shape, self.net1)

        self.net2 = nn.Sequential(
            nn.Linear(*shape_after[1:], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_action),
        )

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


def shape_after_conv(in_shape, net):
    x = torch.randn(1, *in_shape)
    return net(x).shape
