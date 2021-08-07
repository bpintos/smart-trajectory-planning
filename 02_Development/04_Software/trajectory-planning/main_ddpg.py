# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:19:32 2020

@author: bpintos
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import policy,WhiteActionNoise,get_actor,get_critic,Buffer,update_target

np.random.seed(7)
tf.random.set_seed(7)

weights_available = True
weights = 'vehicle_env_continuous_v2_ay_lim_8ms2_obs0'
env_name = 'vehicle_env_continuous'
version = 'v2'

if __name__ == "__main__":
    
    # Environment name for registration
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
    
    env = gym.make(env_name_id)
    
    # Deep deterministic policy gradient taken from https://keras.io/examples/rl/ddpg_pendulum/
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))
    
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    actor_model = get_actor(num_states, upper_bound)
    critic_model = get_critic(num_states, num_actions)
    
    target_actor = get_actor(num_states, upper_bound)
    target_critic = get_critic(num_states, num_actions)
    
    # Initialize weights
    if weights_available == True:
        target_actor.load_weights('vehiclegym/envs/weights/' + weights + '/VehicleTfm_target_actor.h5')
        target_critic.load_weights('vehiclegym/envs/weights/' + weights + '/VehicleTfm_target_critic.h5')
        actor_model.load_weights('vehiclegym/envs/weights/' + weights + '/VehicleTfm_actor.h5')
        critic_model.load_weights('vehiclegym/envs/weights/' + weights + '/VehicleTfm_critic.h5')
    else:
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())
    
    # Learning rate for actor-critic models
    critic_lr = 0.0002 #0.0002
    actor_lr = 0.0001
    
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    
    total_episodes = 20
    # Discount factor for future rewards
    gamma = 0.9 #0.99
    # Used to update target networks 
    tau = 1 #0.02
    # Exploration noise
    std_dev = 0.1
    # Buffer size
    buffer = Buffer(num_states, num_actions, 50000, 64) #5000
    
    # Lists to store episodic rewards and average rewards
    ep_reward_list = []
    avg_reward_list = []
    
    for ep in range(total_episodes):
        prev_state = env.reset()
        episodic_reward = 0
        e_lat_record = np.empty(0)
        e_theta_record = np.empty(0)
        steer_record = np.empty(0)
        steer_raw_record = np.empty(0) 
        steer_noise_record = np.empty(0) 
        reward_record = np.empty(0)
        x_obs_record = np.empty(0)
        y_obs_record = np.empty(0)
        
        fig = plt.figure(figsize=(12, 15))
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax3 = fig.add_subplot(513)
        ax4 = fig.add_subplot(514)
        ax5 = fig.add_subplot(515)
        
        iteration = 0
        
        # Decrease exploration ratio as episodes go on
        if ep <= 40:
            std_dev = 0.3
        elif ep > 40 and ep <= 60:
            std_dev = 0.2
        elif ep > 60 and ep <= 80:
            std_dev = 0.2
        else:
            std_dev = 0.2
        # std_dev = 0.5   
        
        while True:
            env.render()
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            
            action, action_raw, action_noise = policy(tf_prev_state, std_dev, actor_model, lower_bound, upper_bound)
            # action = [np.array(0)]
            
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action, iteration)
            print("Reward ->  {}".format(reward))
            print("Action ->  {}".format(action))
            iteration += 1
            
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            
            buffer.learn(target_actor, target_critic, actor_model, critic_model,\
              actor_optimizer, critic_optimizer, gamma)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            
            e_lat_record = np.append(e_lat_record, state[0])
            e_theta_record = np.append(e_theta_record, state[1])
            x_obs_record = np.append(x_obs_record, state[2])
            y_obs_record = np.append(y_obs_record, state[3])
            steer_record = np.append(steer_record, action)
            steer_raw_record = np.append(steer_raw_record, action_raw)
            steer_noise_record = np.append(steer_noise_record, action_noise)
            reward_record = np.append(reward_record, reward)
            
            # End this episode when `done` is True
            if done:
                break
            
            prev_state = state
        
        ep_reward_list.append(episodic_reward)
        
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
        ax1.plot(e_lat_record)
        ax1.set_title("elat")
        ax2.plot(e_theta_record)
        ax2.set_title("etheta")
        ax3.plot(steer_record, 'b')
        ax3.plot(steer_raw_record, 'r')
        ax3.plot(steer_noise_record, 'g')
        ax3.set_title("delta")
        ax4.plot(x_obs_record, 'b')
        ax4.plot(y_obs_record, 'r')
        ax4.set_title("x and y obstacle")
        ax5.plot(reward_record)
        ax5.set_title("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        plt.show()
        
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
    
    