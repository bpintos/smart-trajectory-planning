# -*- coding: utf-8 -*-
"""
@author: Borja Pintos
"""

# Import necessary packages
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import getQvalue, initQtable_2, getAction, updateQtable
import random

# Make experiments reproducible
np.random.seed(7)
tf.random.set_seed(7)
random.seed(10)

# Beggining of programm
if __name__ == "__main__":
    
    # Give a name to the environment for registration
    env_name = 'diff_env_discrete'
    version = 'v2'
    env_name_id = env_name + '-' + version
    entry_point_name = 'vehiclegym.envs.' + env_name + '_' + version + ':VehicleTfmEnv'
    
    # Calibrate the parameters of the environment accordingly
    config = {
        'x_goal': 19,
        'y_goal': 0,
        'circuit_number': 2,
        'obs': True
    }
    
    # Registry of environment
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if env_name in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    gym.envs.registration.register(
        id = env_name_id,
        entry_point = entry_point_name,
        kwargs = config
    )
    print("Add {} to registry".format(env_name_id))
    
    # Load environment
    env = gym.make(env_name_id)
    
    # Q learning algorithm parameters
    # Learning rate for q learning
    alpha = 1
    # Exploration rate
    epsilon = 0.20
    # Total learning episodes
    total_episodes = 1000
    # Discount factor for future rewards
    gamma = 1
    
    # Variable initialization before episodes
    # Create empty list for reward of each episode
    reward_episode = []
    # Create empty list for average reward over 40 episodes
    avg_reward_list = []
    # Initialize Q table
    q_table = initQtable_2(env_name + '_' + version)
    
    # Q learning algorithm
    for ep in range(total_episodes):
        # Reset environment
        state = env.reset(1000)
        
        # Initialize variables at the beggining of episode
        episodic_reward = 0
        record_state_elat = np.empty(0)
        record_state_long = np.empty(0)
        record_action = np.empty(0)
        record_reward = np.empty(0)
        record_x = np.empty(0)
        record_y = np.empty(0)
        record_theta = np.empty(0)
        
        # Initialize plots
        fig = plt.figure(figsize=(12, 15))
        ax1 = fig.add_subplot(711)
        ax2 = fig.add_subplot(712)
        ax3 = fig.add_subplot(713)
        ax4 = fig.add_subplot(714)
        ax5 = fig.add_subplot(715)
        ax6 = fig.add_subplot(716)
        ax7 = fig.add_subplot(717)
        
        while True:
            # Render environment
            env.render()
            
            # Action determination
            if np.random.uniform(0,1) < epsilon:
                # Exploration
                action = random.choice(q_table[1])
            else:
                # Exploitation
                action = getAction(q_table, state)
            
            # Recieve next state and reward from environment based on the current state and action
            state_next, reward, done, info = env.step([action])
            
            # Acumulate reward over one episode
            episodic_reward += reward
            
            # Update Q value for current state and action
            q_value_old = getQvalue(q_table, state, action)
            action_next = getAction(q_table, state_next)
            q_value_new = getQvalue(q_table, state_next, action_next)
            q_value_new = reward + gamma*q_value_new
            q_value_new = (1-alpha)*q_value_old + alpha*q_value_new
            # Update Q table
            q_table = updateQtable(q_table, state, action, q_value_new)
            
            # Update state
            state = state_next
            
            # Record state
            record_state_elat = np.append(record_state_elat, state[0])
            record_state_long = np.append(record_state_long, state[1])
            record_action = np.append(record_action, action)
            record_reward = np.append(record_reward, reward)
            record_x = np.append(record_x, info[0])
            record_y = np.append(record_y, info[1])
            record_theta = np.append(record_theta, info[2])
            
            # If state is terminal, episode ends
            if done:
                break
        
        # Print episodic reward
        print("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        reward_episode.append(episodic_reward)
        # Mean of last 40 episodes
        avg_reward = np.mean(reward_episode[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
        
        # Plot variables from last episode
        ax1.plot(record_state_elat, color='b', marker="o")
        ax1.set_title("elat")
        ax2.plot(record_state_long, color='b', marker="o")
        ax2.set_title("long_dist")
        ax3.plot(record_action, color='b', marker="o")
        ax3.set_title("action")
        ax4.plot(record_reward, color='b', marker="o")
        ax4.set_title("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        ax5.plot(record_x, color='b', marker="o")
        ax5.set_title("x coordinate")
        ax6.plot(record_y, color='b', marker="o")
        ax6.set_title("y coordinate")
        ax7.plot(record_theta, color='b', marker="o")
        ax7.set_title("theta")
        plt.show()
        
    # Episodes versus Average Reward
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average Episodic Reward")
    plt.show()
    env.close()
    