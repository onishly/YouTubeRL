import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CarRacing-v0'
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

PPO_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'PPO_Model_CarRacing')
model.save(PPO_Path)



