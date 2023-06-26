import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch

env_name = 'CartPole-v0'
env = gym.make(env_name)
'''''
# setting the cartpole environment 5 times - episode is one full game within the environment
# cartpole environment has fixed episode length which is 200 frames whilst others such as CarRacing are continuous
episodes = 5

for episodes in range(1, episodes+1):
    state = env.reset() # env.reset() resets environment and obtains initial environment
    # outputting env.reset() gives us observations from our environment, we can also use env.observation_space.sample() (gives low bound, up bound, no. of values, type)
    done = False # to check whether the episode is done
    score = 0

    while not done:
        env.render() # visualise the environment
        action = env.action_space.sample() #taking action from our discrete action space (here 1 or 0)
        n_state, reward, done, info = env.step(action) # env.step() takes action in the environment. It returns 4 observations, a reward of 1 or 0, and True if episode is done
        score += reward # accumulating our reward
    print("Episode:{} Score: {}".format(episodes, score))
env.close() # close down the render frame
'''''

# Making our directories
log_path = os.path.join('Training', 'Logs')

# TRAINING THE MODEL
env = gym.make(env_name) # Recreating our environment - no need tbh
env = DummyVecEnv([lambda: env]) # allows us to work with our environment that is wrapped inside dummy vectorised environment
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) # Defining our Model / Agent using the dummy vector environment
# we use MlpPolicy - Multi-layer perceptron policy, which is a standard feed forward neural network

model.learn(total_timesteps=20000) # timesteps = training time, you need higher timesteps for the car racing environment, 300-400,000

# SAVING AND RELOADING MODEL
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_path)

# Deleting the RL model
del model

# Reloading the model
model = PPO.load(PPO_path, env=env)
# PPO model in this environment is considered solved when you get a score of 200 and higher

#evaluate = evaluate_policy(model, env, n_eval_episodes=10, render=True)
#print(evaluate)
# out: (200.0,0.0)
# first value is average reward over number of episodes, second value is the standard deviation of these rewards
#env.close()

# TESTING OUR MODEL
episodes = 5

for episodes in range(1, episodes+1):
    obs = env.reset() # env.reset() resets environment and obtains initial environment
    # outputting env.reset() gives us observations from our environment, we can also use env.observation_space.sample() (gives low bound, up bound, no. of values, type)
    done = False # to check whether the episode is done
    score = 0

    while not done:
        env.render() # visualise the environment
        # rather than getting a random action from our observation state through env.action_space.sample(),
        # we use model.predict function on our observations from the environment to generate the next action!
        action, _ = model.predict(obs) # we pass our observations to the model
        obs, reward, done, info = env.step(action) # env.step() takes action in the environment. It returns 4 observations, a reward of 1 or 0, and True if episode is done

        score += reward # accumulating our reward
    print("Episode:{} Score: {}".format(episodes, score))
env.close() # close down the render frame

training_log_path = os.path.join(log_path, 'PPO_26')


# CALL BACK TO THE TRAINING STAGE - brings mean_ep_length and mean_reward which are important metrics
#print("New Model with callback")
save_path = os.path.join('Training','Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
#model.learn(total_timesteps=20000, callback=eval_callback)


# CHANGING POLICY - NEW NEURAL NETWORK by changing number of layers and units
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])] # pi neural network is for custom actor and second one is the value function
                    # 4 layers with 128 units at each layer
# We are still using PPO but we are changing the architecture for the different neural networks used in PPO
print("NEW MODELLLLL WITH NEW NEURAL NETWORKKKK")
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch}) # last part defines a new neural network policy
model.learn(total_timesteps=20000, callback=eval_callback)

# Code to use DQN algorithm instead
#print("\nDQN MODEL!")
#from stable_baselines3 import DQN
#model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
#model.learn(total_timesteps=20000)




