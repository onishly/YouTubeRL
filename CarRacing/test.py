import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CarRacing-v0'
env = gym.make(environment_name)

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))

env.close()