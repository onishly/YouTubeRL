# GYM imports
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from createEnv import ShowerEnv

# Helper imports
import numpy as np
import random
import os

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = ShowerEnv()

# Load model
PPO_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'PPO_Model_ShowerTemp')
model = PPO.load(PPO_Path, env)

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env.close()