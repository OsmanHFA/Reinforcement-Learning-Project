import gym
from gym.wrappers import Monitor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env = gym.make("CarRacing-v0")

ppo_path = os.path.join('Training', 'ZReward Saved Models', 'PPO20_150k_NewTrial_SF50')
model = PPO.load(ppo_path, env=env)

import numpy as np

class PositiveStridesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation.copy()

def evaluate(model, num_episodes=100):
    env = gym.make('CarRacing-v0')
    env = PositiveStridesWrapper(env)
    env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)

    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward}")

    env.close()
    mean_reward = np.mean(episode_rewards)
    print(f"\nMean Reward: {mean_reward:.2f}")


episodes = 10
evaluate(model, episodes)
