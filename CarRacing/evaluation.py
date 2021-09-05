import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
environment_name = 'CarRacing-v0'
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

# Load model
PPO_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'PPO_Model_CarRacing_10M')
model = PPO.load(PPO_Path, env)

# Evaluate model
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

