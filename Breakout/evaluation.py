import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

# Create environment
env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=12)

# Load model
A2C_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'A2C_Model_Breakout_300K')
model = A2C.load(A2C_Path, env)

# Evaluate policy
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

# Close environment
env.close()

