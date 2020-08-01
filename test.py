import unittest

import torch

from train import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_push_pop(self):
        replay_buf = ReplayBuffer(capacity=1000)

        # with bs > 1, need to sort output, but
        # cannot sort bool tensor
        bs = 1

        s = torch.randn(bs, 4, 84, 84)
        a = torch.randint(0, 4, (bs, ))
        r = torch.rand(bs)
        ns = torch.randn(bs, 4, 84, 84)
        d = torch.rand(bs) < 0.5

        for i in range(bs):
            replay_buf.append((s[i], a[i], r[i], ns[i], d[i]))

        s_out, a_out, r_out, ns_out, d_out = replay_buf.sample(bs)
        self.assertTrue(torch.allclose(s.to('cuda:0'), s_out))
        self.assertTrue(torch.allclose(a.to('cuda:0'), a_out))
        self.assertTrue(torch.allclose(r.to('cuda:0'), r_out))
        self.assertTrue(torch.allclose(ns.to('cuda:0'), ns_out))
        self.assertTrue(d.to('cuda:0').eq(d_out).all())


class TestOther(unittest.TestCase):
    def foo(self):
        print("FOOOOO")


if __name__ == '__main__':
    unittest.main()
