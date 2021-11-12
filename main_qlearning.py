# -*- coding: utf-8 -*-
"""
@author: Borja Pintos
"""

# Import necessary packages
import gym
import numpy as np
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import getQvalue, initQtable, getAction, updateQtable
import random
import pickle

# Make experiments reproducible
np.random.seed(7)
random.seed(10)

# Beggining of programm
if __name__ == "__main__":
# =============================================================================
#                            Experiment parameters
# =============================================================================
    # Give a name to the environment for registration
    env_name = 'vehicle_env_discrete' # Environment name
    version = 'v3' # Version
    qtable_comment = 'gamma09_epsilon_02' # q-table saved with this comment
    render = False # set to False to speed up learning
    plot_graphs = False # set to False to speed up learning
    
    # Q learning algorithm parameters
    alpha = 1 # Learning rate for q learning
    epsilon = 0.20 # Exploration rate
    total_episodes = 10000 # Total learning episodes
    gamma = 0.9 # Discount factor for future rewards
    
    # Environment configuration
    circuit_number = 2 # Circuit selection
    starting_pose = [0,0,0] # Starting coordinates. Format: [x,y,heading]
    destination_pose = [19,0,0] # Destination coordinates. Format: [x,y,heading]
    obstacles = True # If True, obstacles are considered
    # Obstacles coordinates. Format [(x1,y1), (x2,y2)]
    obstacles_coor = [(8.0,-0.4),(8.5,-0.4),(9.0,-0.4),(9.5,-0.4),(10.0,-0.4),(10.5,-0.4),(11.0,-0.4),(11.5,-0.4),(12.0,-0.4), \
                      (8.0, 0  ),(8.5, 0  ),(9.0, 0  ),(9.5, 0  ),(10.0, 0  ),(10.5, 0  ),(11.0, 0  ),(11.5, 0  ),(12.0, 0  ), \
                      (8.0, 0.4),(8.5, 0.4),(9.0, 0.4),(9.5, 0.4),(10.0, 0.4),(10.5, 0.4),(11.0, 0.4),(11.5, 0.4),(12.0, 0.4)]
    
    # Sensor discretization
    sensor_discretization = [0.5, 0.4, 5]
    sensor_max = [20, 2, 180]
    sensor_min = [-1, -2, -180]
    
    if version == 'v5':
        observation_discretization = [1, 1, 1]
        observation_max = [5, 4, 4]
        observation_min = [0, 0, 0]
    elif version == 'v4':
        observation_discretization = [0.4]
        observation_max = [2]
        observation_min = [-2]
    elif version == 'v3':
        observation_discretization = [0.5, 0.4]
        observation_max = [20, 2]
        observation_min = [-1, -2]


# =============================================================================
#                     Discretize actions and observations
# =============================================================================
    # Compute discrete actions and observations
    actions_discrete = tuple(range(0,3)) # Actions
    states_discrete = []
    for x, y, z in zip(observation_discretization, observation_max, observation_min):
        states_discrete.append(tuple([round(i,1) for i in np.arange(z,y+x,x)]))
    states_discrete = tuple(states_discrete)
    
    sensor_discrete = []
    for x, y, z in zip(sensor_discretization, sensor_max, sensor_min):
        sensor_discrete.append(tuple([round(i,1) for i in np.arange(z,y+x,x)]))
    sensor_discrete = tuple(sensor_discrete)
    # Ensure that starting and destination poses have a valid discrete value
    starting_pose[0] = min(sensor_discrete[0], key=lambda x: abs(x-starting_pose[0]))
    starting_pose[1] = min(sensor_discrete[1], key=lambda x: abs(x-starting_pose[1]))
    destination_pose[0] = min(sensor_discrete[0], key=lambda x: abs(x-destination_pose[0]))
    destination_pose[1] = min(sensor_discrete[1], key=lambda x: abs(x-destination_pose[1]))
    
    
# =============================================================================
#                        Register and load environment
# =============================================================================
    # Environment name
    env_name_id = env_name + '-' + version
    entry_point_name = 'vehiclegym.envs.' + env_name + '_' + version + ':VehicleTfmEnv'
        
    # Environment configuration
    config = {
        'starting_pose': starting_pose,
        'destination_pose': destination_pose,
        'circuit_number': circuit_number,
        'obs': obstacles,
        'obs_coor': obstacles_coor,
        'sensor_discretization': sensor_discretization,
        'actions': actions_discrete,
        'states': states_discrete
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


# =============================================================================
#                          Variable initialization
# =============================================================================
    reward_episode = [] # Empty list for reward of each episode
    avg_reward_list = [] # Empty list for average reward
    # Initialize Q table
    q_table = initQtable(env_name + '_' + version, states_discrete, actions_discrete)
    # Initial estimation of q-table values
    # q_table[2][::4,1] = 1
    
    
# =============================================================================
#                             Learning process
# =============================================================================
    for ep in range(total_episodes):
        # Reset environment
        state = env.reset(10000)
        
        # Initialize variables at the beggining of episode
        episodic_reward = 0
        iteration = 0
        
        if plot_graphs == True:
            record_action = np.empty(0)
            record_state = []
            for state_var in state:
                record_state.append(np.array(state_var))
            record_reward = np.empty(0)
            record_sensor = [np.empty(0), np.empty(0), np.empty(0)]
            
            # Initialize plots
            fig = plt.figure(figsize=(12, 15))
            ax_action = fig.add_subplot(411)
            ax_state = fig.add_subplot(412)
            ax_reward = fig.add_subplot(413)
            ax_sensor = fig.add_subplot(414)
        
        while True:
            # Render environment
            if render == True:
                env.render()
            
            # Action determination
            if np.random.uniform(0,1) < epsilon:
                # Exploration
                action = random.choice(q_table[1])
            else:
                # Exploitation
                action = getAction(q_table, state)
            
            # action = 1 # Activate only for testing purposes
            # Recieve next state and reward from environment based on the current state and action
            state_next, reward, done, sensors = env.step([action])
            
            # Acumulate reward calculation
            episodic_reward += reward*gamma**iteration
            iteration += 0
            
            # Bellman equation to update q value
            q_value_old = getQvalue(q_table, state, action)
            action_next = getAction(q_table, state_next)
            q_value_new = getQvalue(q_table, state_next, action_next)
            q_value_new = reward + gamma*q_value_new
            # Apply learning rate to update slowly the q value
            q_value_new = (1-alpha)*q_value_old + alpha*q_value_new
            # Update Q table
            q_table = updateQtable(q_table, state, action, q_value_new)
            
            # Update state
            state = state_next
            
            # Record action, state, reward and sensor values
            if plot_graphs == True:
                record_action = np.append(record_action, action)
                for i in range(0,len(state)):
                    record_state[i] = np.append(record_state[i], state[i])
                record_reward = np.append(record_reward, reward)
                for i in range(0,len(sensors)):
                    record_sensor[i] = np.append(record_sensor[i], sensors[i])

            # If state is terminal, episode ends
            if done:
                if render == True:
                    env.render()
                break
        
        # Print episodic reward
        print("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        reward_episode.append(episodic_reward)
        # Mean of last 40 episodes
        avg_reward = np.mean(reward_episode[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
        
        # Plot variables from last episode
        if plot_graphs == True:
            ax_action.plot(record_action, marker="o")
            ax_action.set_title("Action")
            ax_action.set_yticks(np.arange(0, 3, step=1))
            ax_action.grid(color='gray', linestyle='-', linewidth=1)
            for record_state_var in record_state:
                ax_state.plot(record_state_var, marker="o")
            ax_state.set_title("State")
            ax_state.set_yticks(np.arange(0, 6, step=1))
            ax_state.grid(color='gray', linestyle='-', linewidth=1)
            ax_reward.plot(record_reward, marker="o")
            ax_reward.set_title("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
            ax_reward.set_yticks(np.arange(-1, 1.1, step=0.5))
            ax_reward.grid(color='gray', linestyle='-', linewidth=1)
            for record_sensor_var in record_sensor:
                ax_sensor.plot(record_sensor_var, marker="o")
            ax_sensor.set_title("Sensors")
            ax_sensor.set_yticks(np.arange(0, 21, step=5))
            ax_sensor.grid(color='gray', linestyle='-', linewidth=1)
            plt.show()
        
    # Episodes versus Average Reward
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average Episodic Reward")
    plt.show()
    env.close()
    
    # Save q table in a file
    with open('vehiclegym/envs/qtable/' + env_name + '_' + version + '_' + qtable_comment + '.pickle', 'wb') as f:
        pickle.dump(q_table, f)
    