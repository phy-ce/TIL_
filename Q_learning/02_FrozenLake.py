# Hand-written (ほぼサンプルコード)

# Ref :  https://gymnasium.farama.org/environments/toy_text/frozen_lake/

# Action Space
# lef = 0 dow = 1 rig = 2 up = 3 (counter-clock)

# Observation Space
# "The observation is a value representing the player’s current position"
# observation = current_row * ncols(the number of columns) + current_col
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# 12 13 14 15
# start loc = [0,0]

# Starting State
# start loc = [0,0]

# Reward (Default)
# Reach goal : +1
# Reach hole : 0
# Reach frozen : 0

# Episode End
# 1. Termination
# -> The player moves into the hole (Failure)
# -> The player reaches the goal at max(nrow) * max(ncol) - 1 (Observation)
# 2. Truncation (時間制限)
# Episodeが一定数値に達したとき（4x4は100, 8x8は200)

# Arguments

# desc : specify maps ex : desc=["SFFF", "FHFH", "FFFH", "HFFG"] 
# map : predefined map 4x4 & 8x8
# is_slippery = True

"""For example, if action is left, is_slippery is True, and success_rate is 1/3, then:

P(move left)=1/3
P(move up)=1/3
P(move down)=1/3

If action is up, is_slippery is True, and success_rate is 3/4, then:

P(move up)=3/4
P(move left)=1/8
P(move right)=1/8
"""


# Ref : https://huggingface.co/learn/deep-rl-course/unit2/

# reward_schedule : to specify reward amounts for reaching certain tiles.
#================================================
# Reinforcement Learning (RL,強化学習)の目的 : 「累積」報酬の最大化
# 意思決定の過程をpolicy(π)と呼ぶ
# input : 状況
# output : 行動・行動の確率分布

# 最適なpolicyを見つけることが目的
# Policy-Based methods : agentにactionを学習させる
# Value-Based methods : agentにどの状態がよりvalueを持っているか、そのためにはどのactionを取ればいいのか学習させる

# "Value function : Maps a state to the expected value of being at that state"
# input : 状態
# output : πというpolicyを取ったとき期待値

# "Policy-based methods: Directly train the policy to select what action to take given a state
# (or a probability distribution over actions at that state). 
# In this case, we don’t have a value function."
# 
# Policy input : state
# Policy output : action
# (deterministic policy : 状態ごとに行動が決まっている) ↔ stocahstic policy(outputが)
# 

# "Value-based methods: Indirectly, by training a value function that outputs the value of a state or a state-action pair. 
# Given this value function, our policy will take an action."

# argmax_a(Q) = action that makes Q max
# star  = optimal
# Goal : Find Q-table


import numpy as np
import gymnasium as gym
import random
import os
from tqdm import tqdm 
env = gym.make("FrozenLake-v1"
               , map_name="4x4"
               , is_slippery=False
               , render_mode="rgb_array")

# print("_____OBSERVATION SPACE_____ \n")
# print("Observation Space", env.observation_space)
# print("Sample observation", env.observation_space.sample())  # Get a random observation

# input : policy, positive integer num_episode, small positive fraction alpha, GLIE {epsilon_i}
# Output : value function Q (approximately q_pi if num_episodes is large enough)

# initialize Q-table
# rows = states (16)
# columns = actions (4)

def init_Qtable(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])    #index of max value of a certain row(state)
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()  # np.random.randint(0, 4)
    return(action)


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        #repeat
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            new_state, reward, terminated, truncated, info = env.step(action)

            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state][:]) - Qtable[state][action]
            )

            if terminated or truncated:
                break
        
            state = new_state
    
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


sts = env.observation_space.n
acs = env.action_space.n
print("A Num of possible states:", sts)
print("A Num of possible actions:", acs)

Qtable_FL = init_Qtable(sts, acs)

# Training parameters
n_training_episodes = 10000
learning_rate = 0.7

# Evaluation parameters
n_eval_episodes = 100

# Environment parameters
env_id = "FrozenLake-v1"
max_steps = 99  # Max steps per episode
gamma = 0.95    #Discounting rate (for future)
eval_seed = []  #The evaluation seed(?) of the environment

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_FL)


# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


# written by AI

import time

env_render = gym.make("FrozenLake-v1", map_name = "4x4",
                      is_slippery=False, render_mode = "human")

state, _ = env_render.reset()
done = False
while not done:
    action = np.argmax(Qtable_frozenlake[state])
    state, reward, terminated, truncated, _ = env_render.step(action)
    done = terminated or truncated
    time.sleep(0.5)

env_render.close()