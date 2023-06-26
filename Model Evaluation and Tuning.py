from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import os

ppo_path_100k = os.path.join('Training', 'Car Saved Models', 'PPO_100k_Driving_Model')
ppo_path_150k = os.path.join('Training', 'Car Saved Models', 'PPO_150k_Driving_Model')
ppo_path_200k = os.path.join('Training', 'Car Saved Models', 'PPO_200k_Driving_Model')
ppo_path_250k = os.path.join('Training', 'Car Saved Models', 'PPO_250k_Driving_Model')
ppo_path_300k = os.path.join('Training', 'Car Saved Models', 'PPO_300k_Driving_Model')
ppo_path_350k = os.path.join('Training', 'Car Saved Models', 'PPO_350k_Driving_Model')
ppo_path_400k = os.path.join('Training', 'Car Saved Models', 'PPO_400k_Driving_Model')
ppo_path_450k = os.path.join('Training', 'Car Saved Models', 'PPO_450k_Driving_Model')
ppo_path_500k = os.path.join('Training', 'Car Saved Models', 'PPO_500k_Driving_Model')

model_100k = PPO.load(ppo_path_100k)
model_150k = PPO.load(ppo_path_150k)
model_200k = PPO.load(ppo_path_200k)
model_250k = PPO.load(ppo_path_250k)
model_300k = PPO.load(ppo_path_300k)
model_350k = PPO.load(ppo_path_350k)
model_400k = PPO.load(ppo_path_400k)
model_450k = PPO.load(ppo_path_450k)
model_500k = PPO.load(ppo_path_500k)

env = gym.make("CarRacing-v0")

n_eval_episodes = 20  # Number of episodes to run for evaluation

mean_reward_100k, std_reward_100k = evaluate_policy(model_100k, env, n_eval_episodes=n_eval_episodes)
mean_reward_150k, std_reward_150k = evaluate_policy(model_150k, env, n_eval_episodes=n_eval_episodes)
mean_reward_200k, std_reward_200k = evaluate_policy(model_200k, env, n_eval_episodes=n_eval_episodes)
mean_reward_250k, std_reward_250k = evaluate_policy(model_250k, env, n_eval_episodes=n_eval_episodes)
mean_reward_300k, std_reward_300k = evaluate_policy(model_300k, env, n_eval_episodes=n_eval_episodes)
mean_reward_350k, std_reward_350k = evaluate_policy(model_350k, env, n_eval_episodes=n_eval_episodes)
mean_reward_400k, std_reward_400k = evaluate_policy(model_400k, env, n_eval_episodes=n_eval_episodes)
mean_reward_450k, std_reward_450k = evaluate_policy(model_450k, env, n_eval_episodes=n_eval_episodes)
mean_reward_500k, std_reward_500k = evaluate_policy(model_500k, env, n_eval_episodes=n_eval_episodes)

print(f"100k Model: Mean Reward = {mean_reward_100k}, Std. Dev. = {std_reward_100k}")
print(f"150k Model: Mean Reward = {mean_reward_150k}, Std. Dev. = {std_reward_150k}")
print(f"200k Model: Mean Reward = {mean_reward_200k}, Std. Dev. = {std_reward_200k}")
print(f"250k Model: Mean Reward = {mean_reward_250k}, Std. Dev. = {std_reward_250k}")
print(f"300k Model: Mean Reward = {mean_reward_300k}, Std. Dev. = {std_reward_300k}")
print(f"350k Model: Mean Reward = {mean_reward_350k}, Std. Dev. = {std_reward_350k}")
print(f"400k Model: Mean Reward = {mean_reward_400k}, Std. Dev. = {std_reward_400k}")
print(f"450k Model: Mean Reward = {mean_reward_450k}, Std. Dev. = {std_reward_450k}")
print(f"500k Model: Mean Reward = {mean_reward_500k}, Std. Dev. = {std_reward_500k}")















import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [1e-3, 3e-4, 1e-4],
    'n_steps': [128, 256, 512],
    'batch_size': [32, 64, 128],
    'n_epochs': [3, 5, 8, 10, 20],
    'gamma': [0.99, 0.95, 0.90],
    'gae_lambda': [0.99, 0.90, 0.85],
    'clip_range': [0.2, 0.3, 0.4]
}

# Creating the environment
env = gym.make("CarRacing-v0")
env = DummyVecEnv([lambda: env])

# # Loading our model
# ppo_path = os.path.join('Training', 'Car Saved Models', 'PPO_150k_Driving_Model')
# model = PPO.load(ppo_path, env=env)
# model.set_env(env)

# Use GridCV to search for the best hyperparameters
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(env, verbose=1)

# Print best hyperparameters and their score
# print("Best score: {: .2f} using {}".format(grid_search.best_score_, grid_search.best_params_))

# TRAINING and TUNING model
log_path = os.path.join('Training', 'Car Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path, learning_rate=1e-4, n_steps=512, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)
model.learn(total_timesteps=200000)

#  Saving our model
ppo_path = os.path.join('Training', 'Car Saved Models', 'PPO_Another_Tuned_200k_Driving_Model')
model.save(ppo_path)

print("Tuned Model: ")

# TESTING OUR MODEL
episodes = 10

for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward  # accumulating our reward
    print("Episode:{} Score: {}".format(episodes, score))
env.close()  # close down the render frame









