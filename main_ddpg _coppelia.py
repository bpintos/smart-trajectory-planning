# -*- coding: utf-8 -*-
"""
@author: bpintos
"""
import gym
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from trajectoryplan.trajplan import policy, get_NN, Buffer, update_target
import random
import os

# Make experiments reproducible
np.random.seed(7)
tf.random.set_seed(7)
random.seed(123)

if __name__ == "__main__":
# =============================================================================
#                            Experiment parameters
# =============================================================================
    # Environment name
    env_name = 'diff_env_continuous'
    version = 'v1'
    
    # Neural networks parameters
    critic_lr = 0.0002 # Learning rate critic network
    actor_lr = 0.00001 # Learning rate actor network
    actor_hidden_layers = (500,500,500) # Hidden layers of actor NN
    critic_hidden_layers = (500,500,500,500) # Hidden layers of critic NN
    weights = 'center_lane' # Name of the folder which contains the weights
    weights_available = False # Set to True if weights are available at the beggining
    only_simulation = True # Set to true for simulation mode, i.e. no training process (weights must be available)
    
    # DDPG algorithm parameters
    total_episodes = 20 # Total number of learning episodes
    gamma = 0.9 # Discount factor for future rewards
    tau = 1 # Used to update target networks 
    std_dev = 0.5 # Exploration noise
    std_dev_decay = 0.95 # Exploration noise decay
    buffer_capacity = 50000 # Buffer capacity
    batch_size = 64 # Batch size
    
    # Robot parameters
    number_laser = 16 # Number proximity sensors
    number_IMU = 3 # Number variables calculated with IMU sensors (vx, vy, vang)
    number_GPS = 3 # Number variables calculated with GPS sensors (x, y, gamma)
    min_distance2robot = 0.15 # Minimum distance to obstacle to terminate learning episode
    laser_range = 0.6 # Laser maximum range
    wheelBase = 0.331 # Robot's wheel base distance
    wheelRadius = 0.0975 # Robot's wheel radius
    robotLinearVelocity = 0.2 # Robot's linear velocity
    robotMaxLinearVelocity = 1 # Robot's maximum linear velocity
    robotMaxAngularVelocity = 0.5 # Robot's maximum angular velocity
    goal = 4 # Destination coordinates (only x coordinate neccessary for the moment)
    
# =============================================================================
#                        Register and load environment
# =============================================================================
    # Environment name for registration
    env_name_id = env_name + '-' + version
    entry_point_name = 'vehiclegym.envs.' + env_name + '_' + version + ':VehicleTfmEnv'
    
    # Calibrate the parameters of the environment accordingly
    config = {
        'number_laser': number_laser,
        'number_IMU': number_IMU,
        'number_GPS': number_GPS,
        'min_distance2robot': min_distance2robot,
        'laser_range': laser_range,
        'wheelBase': wheelBase,
        'wheelRadius': wheelRadius,
        'robotLinearVelocity': robotLinearVelocity,
        'robotMaxLinearVelocity': robotMaxLinearVelocity,
        'robotMaxAngularVelocity': robotMaxAngularVelocity,
        'goal': goal
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
        actor_target.load_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Actor_target_weights.h5')
        critic_target.load_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Critic_target_weights.h5')
        actor_model.load_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Actor_model_weights.h5')
        critic_model.load_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Critic_model_weights.h5')
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
        
    for ep in range(total_episodes):
        
        print('Episode: ', ep)
        episodic_reward = 0
        state = env.reset()
        sensors_error = [True]*(number_laser + number_IMU + number_GPS)
        actuators_error = [True, True]
        std_dev = std_dev*std_dev_decay
        
        while True:
            # Policy evaluation
            action, action_raw, action_noise = policy(state, std_dev, actor_model, lower_bound, upper_bound)
            # action = 0 # Overwrite action. Only for testing purpuses
            
            # Perform action and observe next state and reward from environment
            state_next, reward, done, sensors_actuators_next_error = env.step(action)
            episodic_reward += reward
            # print ('State:', state_next)
            # print('Action:', action)
            # print('Reward:', reward)
            
            # Record data in buffer
            sensors_next_error = sensors_actuators_next_error[0]
            actuators_next_error = sensors_actuators_next_error[1]
            
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
    
    # Close the connection with Coppelia Sim
    env.closeEnvironment()
    
    # Print episodic reward
    print('Episodic Reward:', episodic_reward)
    
    if not(only_simulation):
        # Create folder
        if not os.path.exists('vehiclegym/weights/' + env_name + '_' + version + '/' + weights):
            os.makedirs('vehiclegym/weights/' + env_name + '_' + version + '/' + weights)
        # Save the weights
        actor_model.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Actor_model_weights.h5')
        critic_model.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Critic_model_weights.h5')
        actor_target.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Actor_target_weights.h5')
        critic_target.save_weights('vehiclegym/weights/' + env_name + '_' + version + '/' + weights + '/Critic_target_weights.h5')
    