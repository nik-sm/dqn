from argparse import ArgumentError


def float01(f):
    f = float(f)
    if f < 0 or f > 1:
        raise ArgumentError('Must be in range [0,1]')
    return f


def toInt(i):
    return int(float(i))
