from collections import deque

from gym import Wrapper


# Rewritten from:
# https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py
class FrameStack(Wrapper):
    def __init__(self, env, num_stack):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        # low = np.repeat(self.observation_space.low[np.newaxis, ...],
        #                 num_stack,
        #                 axis=0)
        # high = np.repeat(self.observation_space.high[np.newaxis, ...],
        #                  num_stack,
        #                  axis=0)
        # self.observation_space = Box(low=low,
        #                              high=high,
        #                              dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.frames.append(observation)
        return list(self.frames), reward, done

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return list(self.frames)
