# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:57:13 2021

@author: bpint
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import policy,WhiteActionNoise,get_actor,get_critic,Buffer,update_target

np.random.seed(7)
tf.random.set_seed(7)

alat_plot = True

if __name__ == "__main__":
    
    # Give a name to the environment for registration
    env_name = 'vehicle_env_continuous'
    version = 'v1'
    env_name_id = env_name + '-' + version
    entry_point_name = 'vehiclegym.envs.' + env_name + '_' + version + ':VehicleTfmEnv'
    
    # Calibrate the parameters of the environment accordingly
    config = {
        'x_goal': 9,
        'y_goal': 0,
        'circuit_number': 2,
        'obs': False
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
    
    std_dev = 0
    # noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    noise = WhiteActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    
    
    actor_model = get_actor(num_states, upper_bound)
    critic_model = get_critic(num_states, num_actions)
    
    target_actor = get_actor(num_states, upper_bound)
    target_critic = get_critic(num_states, num_actions)
    
    # Making the weights equal initially
    target_actor.load_weights('VehicleTfm_target_actor.h5')
    target_critic.load_weights('VehicleTfm_target_critic.h5')
    actor_model.load_weights('VehicleTfm_actor.h5')
    critic_model.load_weights('VehicleTfm_critic.h5')
    
    iteration = 0
    episodic_reward = 0
    prev_state = env.reset()
    
    e_lat_record = np.empty(0)
    e_theta_record = np.empty(0)
    steer_record = np.empty(0)
    steer_raw_record = np.empty(0) 
    steer_noise_record = np.empty(0) 
    reward_record = np.empty(0)
    a_lat_record = np.empty(0)
    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()
        
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        
        action, action_raw, action_noise = policy(tf_prev_state, noise, actor_model, lower_bound, upper_bound, 0)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action, 0)
        print("Reward ->  {}".format(reward))
        print("Action ->  {}".format(action))
        iteration += 1
        # End this episode when `done` is True
        if done:
            env.close()
            break
        prev_state = state
        episodic_reward += reward
        
        e_lat_record = np.append(e_lat_record, state[0])
        e_theta_record = np.append(e_theta_record, state[1])
        steer_record = np.append(steer_record, action)
        steer_raw_record = np.append(steer_raw_record, action_raw)
        steer_noise_record = np.append(steer_noise_record, action_noise)
        reward_record = np.append(reward_record, reward)
        a_lat_record = np.append(a_lat_record, info[3])
    
    if alat_plot:
        fig = plt.figure(figsize=(18, 14))
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax3 = fig.add_subplot(513)
        ax4 = fig.add_subplot(514)
        ax5 = fig.add_subplot(515)
    
        plt.rcParams.update({'font.size': 22})
        time = np.linspace(0, len(e_lat_record)*0.01, len(e_lat_record))
        ax1.plot(time, e_lat_record)
        ax1.set_title("elat [-]")
        ax1.set_yticks(np.arange(-1, 1.1, step=0.5))
        ax1.grid(color='gray', linestyle='-', linewidth=1)
        ax2.plot(time, e_theta_record)
        ax2.set_title("etheta [-]")
        ax2.set_yticks(np.arange(-0.4, 0.5, step=0.2))
        ax2.grid(color='gray', linestyle='-', linewidth=1)
        ax3.plot(time, steer_record)
        ax3.set_title("delta [-]")
        ax3.set_yticks(np.arange(-1, 1.2, step=0.5))
        ax3.grid(color='gray', linestyle='-', linewidth=1)
        ax4.set_title("Lateral acceleration [m/s2]")
        ax4.plot(time, a_lat_record)
        ax4.set_yticks(np.arange(-10, 11, step=5))
        ax4.grid(color='gray', linestyle='-', linewidth=1)
        ax5.set_title("Episodic Cumulative Reward is ==> {}".format(episodic_reward))
        ax5.set_xlabel("Time [s]")
        ax5.plot(time, reward_record)
        ax5.set_yticks(np.arange(0, 1.2, step=0.2))
        ax5.grid(color='gray', linestyle='-', linewidth=1)
        fig.tight_layout(pad=1.0)
        fig.suptitle('Initial conditions: elat_ini = 0.4 m, etheta_ini = 40 deg', fontsize=30, y=1.02)
        plt.show()
    else:
        fig = plt.figure(figsize=(18, 14))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
    
        plt.rcParams.update({'font.size': 22})
        time = np.linspace(0, len(e_lat_record)*0.01, len(e_lat_record))
        ax1.plot(time, e_lat_record)
        ax1.set_title("elat [-]")
        ax1.set_yticks(np.arange(-1, 1.1, step=0.5))
        ax1.grid(color='gray', linestyle='-', linewidth=1)
        ax2.plot(time, e_theta_record)
        ax2.set_title("etheta [-]")
        ax2.set_yticks(np.arange(-0.4, 0.5, step=0.2))
        ax2.grid(color='gray', linestyle='-', linewidth=1)
        ax3.plot(time, steer_record)
        ax3.set_title("delta [-]")
        ax3.set_yticks(np.arange(-1, 1.2, step=0.5))
        ax3.grid(color='gray', linestyle='-', linewidth=1)
        ax4.set_title("Episodic Cumulative Reward is ==> {}".format(episodic_reward))
        ax4.set_xlabel("Time [s]")
        ax4.plot(time, reward_record)
        ax4.set_yticks(np.arange(0, 1.2, step=0.2))
        ax4.grid(color='gray', linestyle='-', linewidth=1)
        fig.tight_layout(pad=1.0)
        fig.suptitle('Initial conditions: elat_ini = -0.4 m, etheta_ini = -40 deg', fontsize=30, y=1.02)
        plt.show()