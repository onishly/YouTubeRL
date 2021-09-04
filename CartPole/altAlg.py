import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')

# Load environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)
# Wrap environment inside DummyVecEnv wrapper
# Created a lambda function (environment creation function)
# This allows us to work with a vectorised environment
# It works as a wrapper for a non-vectorised environment
env = DummyVecEnv([lambda: env])

# Define new neural network architecture
# `pi` is the neural network architecture for our custom actor, it has 4 layers with 128 nodes in each layer
# `vf` is te neural network architecture for the value function, it has 4 layers with 128 nodes in each layer
net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

# Create PPO model
# Set policy to Multi-layer perceptron policy, meaning we're using a standard NN without LSTM or CNN layers
# Pass through environment
# verbose=1 to log results for model
# Specify tensorboard log path
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# Set save path for best model
best_model_save_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models')
# Set callbacks
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
# This is the callback that is going to be triggered after each training run
# Every time there is a new best model, stop_callback will be triggered
# If stop_callback realises that the reward is above 200 then it stops training
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=best_model_save_path,
                             verbose=1)

model.learn(total_timesteps=20000)
