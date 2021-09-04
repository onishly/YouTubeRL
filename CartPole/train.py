import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')

# Load Environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)  # Load environment
# Wrap environment inside DummyVecEnv wrapper
# Created a lambda function (environment creation function)
# This allows us to work with a vectorised environment
# It works as a wrapper for a non-vectorised environment
env = DummyVecEnv([lambda: env])
# Create PPO model
# Set policy to Multi-layer perceptron policy, meaning we're using a standard NN without LSTM or CNN layers
# Pass through environment
# verbose=1 to log results for model
# Specify tensorboard log path
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# Train model with specified number of timesteps
model.learn(total_timesteps=20000)

# Save model
PPO_Path = os.path.join('../Training', 'Saved Models', 'PPO_Model_CartPole')
model.save(PPO_Path)


