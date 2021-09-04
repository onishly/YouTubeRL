import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load Environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5
for episode in range (1, episodes+1):
    state = env.reset()  # Gives observations for our particular pole
    done = False
    score = 0

    while not done:
        env.render()  # Renders environment
        action = env.action_space.sample()  # Generate random action, can only be 0 or 1 as action_space is Discrete(2)
        # Pass through random action to environment, we get back:
        # Next set of observations
        # Reward
        # Whether episode is done or not
        n_state, reward, done, info = env.step(action)
        score += reward  # Accumulate reward

    print('Episode: {} Score: {}'.format(episode, score))

env.close()

# Understanding the environment
# env.observation_space gives 4 values: cart position, cart velocity, pole angle, and pole angular velocity
# env.action_space has two possible discrete values; push cart to the left and push cart to the right
