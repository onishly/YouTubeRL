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

# Create model
log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# Start learning
model.learn(total_timesteps=100000)

# Save model
PPO_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'PPO_Model_ShowerTemp')
model.save(PPO_Path)