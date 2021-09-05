import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

environment_name = 'CarRacing-v0'
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=10)

log_path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Logs')
model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log=log_path)
PPO_Path = os.path.join(os.path.dirname(os.getcwd()), 'Training', 'Saved Models', 'PPO_Model_CarRacing_10M')
# model = PPO.load(PPO_Path, env)
model.learn(total_timesteps=10000000)


model.save(PPO_Path)



