# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:57:13 2021

@author: bpint
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectoryplan.trajplan import getQvalue, initQtable, getAction, updateQtable
import random
import pickle
import time

# with open('train6.pickle', 'wb') as f:
#     pickle.dump(q_table, f)


with open('train5.pickle', 'rb') as f:
    q_table = pickle.load(f)

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
state = env.reset(1000)
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
    time.sleep(0.8)
    states.append(state)
    rewards.append(reward)
    infos.append(info)
    if done:
        break

env.close()
print("Episodic Reward is ==> {}".format(episodic_reward))