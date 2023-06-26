import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env = gym.make("CarRacing-v0")
#  Reward is -0.1 for every frame and +1000/N for every track tile visited, where N is the total number of tiles visited
#  e.g. finished 732 frames, reward is 1000 - 0.1*732 = 926.8 points
#  environment solved when agent consistently gets 900+ points, track is random every episode
#  Before PPO Implementation: car is taking random actions and does not really know the track

#  Training our Model
env = DummyVecEnv([lambda: env])
log_path = os.path.join('Training', 'Car Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=50000)

#  Saving our model
ppo_path = os.path.join('Training', 'Car Saved Models', 'PPO_50k_Driving_Model')
model.save(ppo_path)
print("Model Saved!")

#  Evaluating and Testing Model (optional)
#  evaluate = evaluate_policy(model, env, n_eval_episodes=10, render=True)
#  print("Policy Evaluation: ", evaluate)

print("Now let us take 15 episodes using normal method: ")
# TESTING OUR MODEL
episodes = 15

for episodes in range(1, episodes+1):
    obs = env.reset()  # env.reset() resets environment and obtains initial environment
    # outputting env.reset() gives us observations from our environment, we can also use env.observation_space.sample() (gives low bound, up bound, no. of values, type)
    done = False # to check whether the episode is done
    score = 0

    while not done:
        env.render() # visualise the environment
        # rather than getting a random action from our observation state through env.action_space.sample(),
        # we use model.predict function on our observations from the environment to generate the next action!
        action, _ = model.predict(obs)  # we pass our observations to the model
        obs, reward, done, info = env.step(action)  # env.step() takes action in the environment. It returns 4 observations, a reward of 1 or 0, and True if episode is done

        score += reward  # accumulating our reward
    print("Episode:{} Score: {}".format(episodes, score))
env.close()  # close down the render frame

