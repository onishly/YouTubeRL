# GYM imports
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Helper imports
import numpy as np
import random
import os

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        # Apply temp adj
        self.state += action - 1

        # Decrease shower time
        self.shower_length -= 1

        # Calculate reward
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        # TODO: Implement viz
        pass

    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60

        return self.state
