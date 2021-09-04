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

# Rather than taking random steps, use the model to choose a step
episodes = 5
for episode in range (1, episodes+1):
    obs = env.reset()  # Gives observations for our particular pole
    done = False
    score = 0

    while not done:
        env.render()  # Renders environment
        action, _ = model.predict(obs)  # Uses trained model to pick action
        # Pass through random action to environment, we get back:
        # Next set of observations
        # Reward
        # Whether episode is done or not
        obs, reward, done, info = env.step(action)
        score += reward  # Accumulate reward

    print('Episode: {} Score: {}'.format(episode, score))

env.close()