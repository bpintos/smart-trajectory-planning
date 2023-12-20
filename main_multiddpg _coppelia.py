# -*- coding: utf-8 -*-
"""
@author: bpintos
"""
# Import packages
import gym
import numpy as np
import tensorflow as tf
from trajectoryplan.trajplan import policy, get_NN, Buffer, update_target
import random
import os
import time
# from tools.support_tools import plot_actor, plot_critic
from subprocess import Popen
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import ast

# Make experiments reproducible
np.random.seed(7)
tf.random.set_seed(7)
random.seed(123)

# Start time
start = time.time()

# Main script
if __name__ == "__main__":
# =============================================================================
#                                Load XML file
# =============================================================================
    param_file = 'param11_ddpg.xml'
    tree = ET.parse('vehiclegym/param/'+ param_file)
    root = tree.getroot()
# =============================================================================
#                        Read parameters from XML file
# =============================================================================
    # Environment name
    vehicle = root[0][0].text # Vehicle type
    env_name = root[0][1].text # Environment name
    version = root[0][2].text # Environment version
    only_simulation = root[0][3].text == 'True' # Set to true for simulation mode, i.e. no training process (weights must be available)
    destinations = ast.literal_eval(root[0][4].text) # Destination coordinates (only x coordinate neccessary for the moment)
    
    # DDPG algorithm parameters
    total_episodes = int(root[1][0].text) # Total number of learning episodes
    gamma = float(root[1][1].text) # Discount factor for future rewards
    tau = float(root[1][2].text) # Used to update target networks 
    std_dev = np.array(ast.literal_eval(root[1][3].text)) # Exploration noise
    std_dev_decay = float(root[1][4].text) # Exploration noise decay
    buffer_capacity = int(root[1][5].text) # Buffer capacity
    batch_size = int(root[1][6].text) # Batch size
    critic_lr = float(root[1][7].text) # Learning rate critic network
    actor_lr = float(root[1][8].text) # Learning rate actor network
    actor_hidden_layers = ast.literal_eval(root[1][9].text) # Hidden layers of actor NN
    critic_hidden_layers = ast.literal_eval(root[1][10].text) # Hidden layers of critic NN
    weights_input = root[1][11].text # Read in NN weights if available
    weights_available = root[1][12].text == 'True' # Set to True if weights are available at the beggining
    weights_output = root[1][13].text # Write NN weights after learning process
    
    # Robot parameters
    number_proximity = int(root[2][0].text) # Number proximity sensors
    number_laser = int(root[2][1].text) # Number laser sensors
    number_IMU = int(root[2][2].text) # Number variables calculated with IMU sensors (vx, vy, vang)
    number_GPS = int(root[2][3].text) # Number variables calculated with GPS sensors (x, y, gamma)
    number_laser_rays = int(root[2][4].text) # Number of laser rays considered in the laser sensor
    laser_range = float(root[2][5].text) # Laser maximum range
    wheelBase = float(root[2][6].text) # Robot's wheel base distance
    wheelRadius = float(root[2][7].text) # Robot's wheel radius
    robotLinearVelocity = float(root[2][8].text) # Robot's linear velocity
    robotMaxLinearVelocity = float(root[2][9].text) # Robot's maximum linear velocity
    robotMaxAngularVelocity = float(root[2][10].text) # Robot's maximum angular velocity
    min_distance2robot = float(root[2][11].text) # Minimum distance to obstacle to terminate learning episode
    
    # Coppelia scene and options
    scene = root[3][0].text
    coppelia_GUI = root[3][1].text == 'True'
    
    # Online learning
    onlineLearning = False
# =============================================================================
#                        Register and load environment
# =============================================================================
    # Start up CoppeliaSim
    if coppelia_GUI:
        mode = ''
    else:
        mode = '-h'
    arguments = [mode, scene]
    Popen(['python3', 'coppelia/startupCoppelia.py'] + arguments)
    if coppelia_GUI:
        time.sleep(10)
    else:
        time.sleep(2)
# =============================================================================
#                        Register and load environment
# =============================================================================
    # Environment name for registration
    env_name_id = vehicle + '_' + env_name + '-' + version
    entry_point_name = 'vehiclegym.envs.vehicle.' + vehicle + '.' + env_name + '_' + version + ':VehicleTfmEnv'
    
    # Calibrate the parameters of the environment accordingly
    config = {
        'number_proximity': number_proximity,
        'number_laser': number_laser,
        'number_IMU': number_IMU,
        'number_GPS': number_GPS,
        'number_laser_rays': number_laser_rays,
        'min_distance2robot': min_distance2robot,
        'laser_range': laser_range,
        'wheelBase': wheelBase,
        'wheelRadius': wheelRadius,
        'robotLinearVelocity': robotLinearVelocity,
        'robotMaxLinearVelocity': robotMaxLinearVelocity,
        'robotMaxAngularVelocity': robotMaxAngularVelocity
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
# =============================================================================
#                              DDPG algorithm
# ============================================================================= 
    # Size of state and action spaces    
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))
    
    # Limits of the action space
    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    # Creation of actor and critic neural networks
    actor_model = get_NN(num_states, actor_hidden_layers, num_actions, hidden_activation = 'relu', output_activation = 'tanh')
    critic_model = get_NN(num_states+num_actions, actor_hidden_layers, 1, hidden_activation = 'relu')
    actor_target = get_NN(num_states, actor_hidden_layers, num_actions, hidden_activation = 'relu', output_activation = 'tanh')
    critic_target = get_NN(num_states+num_actions, actor_hidden_layers, 1, hidden_activation = 'relu')
    
    # Initialize weights
    if weights_available or only_simulation:
        actor_target.load_weights('vehiclegym/weights/' + vehicle + '/' + env_name + '_' + version + '/' + weights_input + '/Actor_target_weights.h5')
        critic_target.load_weights('vehiclegym/weights/' + vehicle + '/' + env_name + '_' + version + '/' + weights_input + '/Critic_target_weights.h5')
        actor_model.load_weights('vehiclegym/weights/' + vehicle + '/' + env_name + '_' + version + '/' + weights_input + '/Actor_model_weights.h5')
        critic_model.load_weights('vehiclegym/weights/' + vehicle + '/' + env_name + '_' + version + '/' + weights_input + '/Critic_model_weights.h5')
    else:
        actor_target.set_weights(actor_model.get_weights())
        critic_target.set_weights(critic_model.get_weights())
    
    # Neural network optimizer configuration
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    
    # Buffer size
    buffer = Buffer(num_states, num_actions, buffer_capacity, batch_size)    
# =============================================================================
#                          DDPG learning process
# =============================================================================
    # If simulation flag is enabled, set total number of episodes to 1
    # and exploration ratio to 0
    if only_simulation:
        total_episodes = 1
        std_dev = 0
    
    # Empty variables to record data
    action_record = []
    state_record = []
    sensors_record = []
    reward_record = []
    episodic_reward_record = []
    firstRun = True
    destination_reached = False
        
    for ep in range(total_episodes):
        
        print('Episode: ', ep)
        episodic_reward = 0
        num_iter = 0
        sensors_error = [True]*(number_laser + number_IMU + number_GPS)
        actuators_error = [True, True]
        std_dev = std_dev*std_dev_decay
        
        if onlineLearning:
            if firstRun:
                state = env.reset()
                env.setTargetDestination(firstRun, destinations)
                firstRun = False
            else:
                if destination_reached:
                    env.setTargetDestination(firstRun, destinations)
                print('Backing up and Turning')
                for _ in range(8):
                    state, _, _ , _ = env.step(np.array([-0.2,0]))
                for _ in range(20):
                    state, _, _ , _ = env.step(np.array([0,1]))
                print('Manouvring Completed')
        else:
            if firstRun:
                state = env.reset()
                env.setTargetDestination(firstRun, destinations)
                firstRun = False
            else:
                if destination_reached:
                    env.setTargetDestination(firstRun, destinations)
                    state = env.reset()
                else:
                    state = env.reset()
        
        while True:
            # Policy evaluation
            action, action_raw, action_noise = policy(state, std_dev, actor_model, lower_bound, upper_bound, num_iter)
            # action = [0.08, 0]# Overwrite action. Only for testing purpuses
            
            # Perform action and observe next state and reward from environment
            state_next, reward, done, info = env.step(action)
            # state_next, reward, done, info = env.step(np.array([0.2, 0])) # Overwrite action. Only for testing purpuses
            episodic_reward += reward
            # print ('State:', state_next)
            # print('Action:', action)
            # print('Reward:', reward)
            
            # Record data in buffer
            sensors_next_error = info[0]
            actuators_next_error = info[1]
            sensors = info[2]
            counter_done = info[3]
            destination_reached = info[4]
            
            # Learning process with replay buffer
            if not(only_simulation):
                # Check if exist any error in sensors or actuators
                if all(sensor_next_error == False for sensor_next_error in sensors_next_error) and \
                    all(actuator_next_error == False for actuator_next_error in actuators_next_error) and \
                        all(sensor_error == False for sensor_error in sensors_error) and \
                            all(actuator_error == False for actuator_error in actuators_error):
                    buffer.record((state, action, reward, state_next))
                    
                # Update neural networks with Bellman equation
                buffer.learn(actor_target, critic_target, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma)
                    
                # Update target neural networks
                update_target(actor_target.variables, actor_model.variables, tau)
                update_target(critic_target.variables, critic_model.variables, tau)
            
            # If state_next is terminal, episode ends
            if done:
                break
            
            # Update old state and errors
            state = state_next
            sensors_error = sensors_next_error
            actuators_error = actuators_next_error
            num_iter += 1
            
            # Print mean episodic reward
            # print('Episodic Reward:', episodic_reward/num_iter)
            
            # Record data
            if only_simulation:
                action_record.append(action)
                state_record.append(state)
                sensors_record.append(sensors)
                reward_record.append(reward)
        
        # Print counter
        print('Episode: ', ep)
        print('counter_done:', counter_done)
        episodic_reward_record.append(episodic_reward/num_iter)
        if counter_done >= 50 and ep >= 50:
            break
    
    # Close the connection with Coppelia Sim
    env.closeEnvironment()
    
    # Data post processing
    if not(only_simulation):
        # Save the weights of the models
        if not os.path.exists('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output):
            os.makedirs('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output)
        actor_model.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output + '/Actor_model_weights.h5')
        critic_model.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output + '/Critic_model_weights.h5')
        actor_target.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output + '/Actor_target_weights.h5')
        critic_target.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights_output + '/Critic_target_weights.h5')
    else:
        # Plot main variables
        fig = plt.figure(figsize=(18, 14))
        fontsize = 24
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        plt.rcParams.update({'font.size': 22})
        time_plot = np.linspace(0, len(action_record)*0.2, len(action_record))
        # vel_ang = list(map(lambda x: x[1], action_record))
        ax1.plot(time_plot, list(map(lambda x: x[0], action_record)), 'b')
        ax1.plot(time_plot, list(map(lambda x: x[1], action_record)), 'r')
        ax1.set_title('Actuator signals over time: Normalized linear (blue) and angular (red) velocities', fontsize=fontsize)
        # ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Normalized velocity [-]', fontsize=fontsize)
        ax1.set_yticks(np.arange(-1, 1.1, step=0.2))
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        ax1.set_xticklabels([])
        ax1.grid(color='gray', linestyle='-', linewidth=1)
        vel_x_record = list(map(lambda x: x[16], sensors_record))
        vel_y_record = list(map(lambda x: x[17], sensors_record))
        vel_lin_record = list(map(lambda x, y: np.sqrt(x**2 + y**2), vel_x_record, vel_y_record))
        vel_ang_record = list(map(lambda x: x[18], sensors_record))
        ax2.plot(time_plot, vel_lin_record, 'b')
        ax2.plot(time_plot, vel_ang_record, 'r')
        ax2.set_title('Sensor signals over time: Linear (blue) and angular (red) velocities', fontsize=fontsize)
        # ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Velocity [m/s, rad/s]', fontsize=fontsize)
        ax2.set_yticks(np.arange(-0.6, 0.65, step=0.2))
        ax2.grid(color='gray', linestyle='-', linewidth=1)
        accel_ang_record = list(map(lambda x: x[19], sensors_record))
        accel_lin_record = list(map(lambda x: x[20], sensors_record))
        ax3.plot(time_plot, accel_lin_record, 'b')
        ax3.plot(time_plot, accel_ang_record, 'r')
        ax3.set_title('Evolution of linear and angular acceleration over time', fontsize=fontsize)
        ax3.set_xlabel('Time [s]', fontsize=fontsize)
        ax3.set_ylabel('Acceleration [m/s2, rad/s2]', fontsize=fontsize)
        ax3.set_yticks(np.arange(-2, 2.6, step=0.5))
        ax3.grid(color='gray', linestyle='-', linewidth=1)
        fig.tight_layout(pad=1.0)
        # fig.suptitle('', fontsize=30, y=1.02)
        plt.show()
    
    # Plot the actor model
    # title = 'Actor model'
    # plot_actor(title, actor_model, env.observation_space.high, env.observation_space.low)
    
    # Plot the target actor model
    # title = 'Target actor model'
    # plot_actor(title, actor_target, env.observation_space.high, env.observation_space.low)
    
    # Plot the critic model
    # title = 'Critic model'
    # plot_critic(title, critic_model, env.observation_space.high, env.observation_space.low,\
    #             env.action_space.high, env.action_space.low)
    
    # Plot the target critic model
    # title = 'Target critic model'
    # plot_critic(title, critic_target, env.observation_space.high, env.observation_space.low,\
    #             env.action_space.high, env.action_space.low)
    
    # Plot episodic reward
    # fig = plt.figure(figsize=(18, 14))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(episodic_reward_record_04, 'b', label='Single action, safety req.')
    # ax1.plot(episodic_reward_record_05, 'r', label='Multiple actions, safety+legal req.')
    # ax1.plot(episodic_reward_record_06, 'g', label='Multiple actions, safety+legal+comfort req.')
    # ax1.plot(episodic_reward_record_07, 'k', label='Multiple actions, safety+legal+comfort req., stop strategy')
    # ax1.set_yticks(np.arange(-1.5, 0.1, step=0.5))
    # ax1.grid(color='gray', linestyle='-', linewidth=1)
    # ax1.set_title('Episodic reward')
    # ax1.set_xlabel('Episode number')
    # ax1.set_ylabel('Episodic reward')
    # ax1.legend()
    # plt.show()
    
    # Compute elapsed time and print it
    end = time.time()
    print('Elapsed Time:', end - start)