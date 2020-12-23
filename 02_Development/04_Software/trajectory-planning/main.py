# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:19:32 2020

@author: bpint
"""
import gym
import vehiclegym

if __name__ == "__main__":
    env_name = 'VehicleTfm-v0'
    config = {
        'x_goal': 5,
        'y_goal': 0,
        'left_lane': -0.5,
        'right_lane': 0.5,
    }
    
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
    
    for _ in range(100):
        env.render()
        env.step([2])
        print(env._reward())
    env.close()