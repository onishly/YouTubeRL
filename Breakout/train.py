import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

# Load 4 different environments by vectorising environments
env = make_atari_env('Breakout-v0', n_envs=12, seed=0)
env = VecFrameStack(env, n_stack=12)

# Create model
log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# Learn
model.learn(total_timesteps=300000)
A2C_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'A2C_Model_Breakout_300K')
model.save(A2C_Path)