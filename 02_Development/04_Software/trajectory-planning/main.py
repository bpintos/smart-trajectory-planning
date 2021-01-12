# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:19:32 2020

@author: bpintos
"""
import gym
import vehiclegym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import policy,OUActionNoise,get_actor,get_critic,Buffer,update_target

if __name__ == "__main__":
    env_name = 'VehicleTfm-v0'
    config = {
        'x_goal': -1,
        'y_goal': 0,
        'circuit_number': 1
    }
    
    # Registry of environment
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if env_name in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    gym.envs.registration.register(
        id=env_name,
        entry_point='vehiclegym.envs.vehicle_env:VehicleTfmEnv',
        kwargs=config
    )
    print("Add {} to registry".format(env_name))
    
    env = gym.make(env_name)
    
    # Deep deterministic policy gradient taken from https://keras.io/examples/rl/ddpg_pendulum/
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))
    
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    
    actor_model = get_actor(num_states, upper_bound)
    critic_model = get_critic(num_states, num_actions)
    
    target_actor = get_actor(num_states, upper_bound)
    target_critic = get_critic(num_states, num_actions)
    
    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    
    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    
    total_episodes = 100
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005
    
    buffer = Buffer(num_states, num_actions, 50000, 64)
    
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    
    # Takes about 4 min to
    for ep in range(total_episodes):
        prev_state = env.reset()
        episodic_reward = 0
        
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            env.render()
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            
            action = policy(tf_prev_state, ou_noise, actor_model, lower_bound, upper_bound)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
            print("Reward ->  {}".format(reward))
            print("Action ->  {}".format(action))
            
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            
            buffer.learn(target_actor, target_critic, actor_model, critic_model,\
              actor_optimizer, critic_optimizer, gamma)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            
            # End this episode when `done` is True
            if done:
                break
            
            prev_state = state
        
        ep_reward_list.append(episodic_reward)
        
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
        
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    env.close()
    
    # Save the weights
    actor_model.save_weights("VehicleTfm_actor.h5")
    critic_model.save_weights("VehicleTfm_critic.h5")
    
    target_actor.save_weights("VehicleTfm_target_actor.h5")
    target_critic.save_weights("VehicleTfm_target_critic.h5")
    
    