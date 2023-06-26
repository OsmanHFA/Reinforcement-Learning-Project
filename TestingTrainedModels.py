import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


env = gym.make("CarRacing-v0")
env = DummyVecEnv([lambda: env])

ppo_path = os.path.join('Training', 'Car Saved Models', 'PPO_50k_Driving_Model')
model = PPO.load(ppo_path, env=env)
model.set_env(env)

# TESTING OUR MODEL
episodes = 15

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

#  Evaluating and Testing Model

#evaluate = evaluate_policy(model, env, n_eval_episodes=15, render=True)
# print("Policy Evaluation: ", evaluate)
