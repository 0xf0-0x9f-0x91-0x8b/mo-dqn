import numpy as np
import gymnasium as gym


class LinearScalarization(gym.Wrapper):

    def __init__(self, env, weights):
        super(LinearScalarization, self).__init__(env)
        self.weights = weights
        self.mo_return = None

    def reset(self, seed, options):
        o = super(LinearScalarization, self).reset()
        self.mo_return = 0.
        return o

    def step(self, a):
        o, r, d, t, i = super(LinearScalarization, self).step(a)
        self.mo_return += r
        scalarized = np.dot(r, self.weights)
        return o, scalarized, d, t, i