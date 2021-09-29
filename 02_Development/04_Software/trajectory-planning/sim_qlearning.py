# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:57:13 2021

@author: bpint
"""
import gym
import numpy as np
from trajectoryplan.trajplan import getAction
import pickle
import time

# =============================================================================
#                            Experiment parameters
# =============================================================================
# Give a name to the environment for registration
env_name = 'vehicle_env_discrete' # Environment name
version = 'v3' # Version
qtable_comment = 'gamma09_epsilon_02' # q-table saved with this comment
render = True # set to False to speed up learning
    
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
#                              Load Q table
# =============================================================================
with open('vehiclegym/envs/qtable/' + env_name + '_' + version + '_' + qtable_comment + '.pickle', 'rb') as f:
    q_table = pickle.load(f)


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
    'obs_coor':obstacles_coor,
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
#                               Simulation
# =============================================================================
state = env.reset(10000)
episodic_reward = 0
states = [state]
rewards = []
infos =  []

while True:
    # Render environment
    env.render()
    action = getAction(q_table, state)

    # Recieve next state and reward from environment based on the current state and action
    state_next, reward, done, info = env.step([action])
    # If state is terminal, episode ends
    state = state_next
    episodic_reward += reward
    time.sleep(0.4)
    states.append(state)
    rewards.append(reward)
    infos.append(info)
    if done:
        env.render()
        break

env.close()
print("Episodic Reward is ==> {}".format(episodic_reward))