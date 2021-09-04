import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)  # Load environment
env = DummyVecEnv([lambda: env])

# Load model
PPO_Path = os.path.join('../Training', 'Saved Models', 'PPO_Model_CartPole')
model = PPO.load(PPO_Path, env=env)

# Evaluate policy
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env.close()
