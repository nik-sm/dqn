from argparse import ArgumentError

import torch


def float01(f):
    f = float(f)
    if f < 0 or f > 1:
        raise ArgumentError('Must be in range [0,1]')
    return f


def toInt(i):
    return int(float(i))


def huber(input, target, delta=1.):
    t = torch.abs(input - target)
    return torch.mean(
        torch.where(t <= delta, 0.5 * t**2, t * delta - (0.5 * delta**2)))
