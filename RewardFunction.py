import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
from gym import spaces

class CustomCarRacingEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        modified_reward = self.custom_reward_function(observation, action, reward)
        observation = observation.copy()  # Add this line
        return observation, modified_reward, done, info

    def reset(self):
        return self.env.reset()

    def custom_reward_function(self, observation, action):
        # Implement your alternative reward function here
        def custom_reward_function(self, observation, action, original_reward):
            steering = action[0]

            # Penalize sudden steering changes
            if self.prev_steering is not None:
                steering_change = abs(steering - self.prev_steering)
                steering_penalty = steering_change * 50  # ADJUSTING SCALING FACTOR
                modified_reward = original_reward - steering_penalty
            else:
                modified_reward = original_reward

            self.prev_steering = steering

            return modified_reward

# Usage:
env = gym.make("CarRacing-v0")
custom_env = CustomCarRacingEnv(env)


# Train our model using PPO
log_path = os.path.join('Training', 'ZReward Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=150000)

# #  Saving our model
ppo_path = os.path.join('Training', 'ZReward Saved Models', 'PPO20_150k_NewTrial_SF50')
model.save(ppo_path)

# TESTING OUR MODEL
episodes = 15

for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs.copy())
        obs, reward, done, info = env.step(action)
        score += reward  # accumulating our reward
    print("Episode:{} Score: {}".format(episodes, score))
env.close()  # close down the render frame
