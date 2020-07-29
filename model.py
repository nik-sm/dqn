import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_action=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),  # TODO - padding?
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),  # TODO - padding?
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        shape_after = shape_after_conv((4, 84, 84), self.net1)

        self.net2 = nn.Sequential(
            nn.Linear(shape_after, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_action),
        )

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


def shape_after_conv(in_shape, net):
    x = torch.randn(1, *in_shape).to(net.device)
    return net(x).shape.squeeze()
