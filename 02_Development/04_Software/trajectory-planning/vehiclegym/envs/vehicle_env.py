# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:36:31 2020

@author: bpintos
"""

import gym
from gym import spaces
import numpy as np
from pyfmi import load_fmu
import time

class VehicleTfmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, x_goal, y_goal, left_lane, right_lane):
        super(VehicleTfmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.max_steering = 20
        high = np.array([self.max_steering], dtype = np.float32)
        self.action_space = spaces.Box(-high, high)
        high = np.array([np.inf, np.inf, 360], dtype = np.float32)
        low = np.array([-np.inf, -np.inf, 0], dtype = np.float32)
        self.observation_space = spaces.Box(low, high)
        
        model_path = 'D:/Projects/MasterIA/TFM/smart-trajectory-planning/02_Development/02_Libraries/Chasis_simpl.fmu'
        self.model = load_fmu(model_path) 
        
        self.sample_time = 0.01
        self.start_time = 0
        self.stop_time = self.sample_time
        self.done = False
        self.state = self.reset()
    
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.left_lane = left_lane
        self.right_lane = right_lane
    
        self.viewer = None
        self.display = None
    
        self.veh_transform = None

    def step(self, action):
        # Execute one time step within the environment
        self.model.set(list(['delta']), list(action))
        opts = self.model.simulate_options()
        opts['ncp'] = 10
        opts['initialize'] = False
        res = self.model.simulate(start_time=self.start_time, final_time=self.stop_time, options=opts)
        self.state = tuple([res.final(k) for k in ['x','y','theta_out']])
    
        self.done = self._is_done()
        
        if not self.done:
            self.start_time = self.stop_time
            self.stop_time += self.sample_time
    
        return self.state, self._reward(), self.done, {}
     
    def reset(self):
        # Reset the state of the environment to an initial state
        self.model.reset()
        res = self.model.simulate(start_time=0, final_time=0)
        
        return tuple([res.final(k) for k in ['x','y','theta_out']])
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return True
        screen_width = 600
        screen_height = 900
        
        img_x_orig = 300
        img_y_orig = 0
        
        m2pixel = 200
        
        veh_width = 0.2
        veh_height = 0.4
        veh_width = veh_width*m2pixel
        veh_height = veh_height*m2pixel
        
        lane_width = 1
        lane_width = lane_width*m2pixel
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            # add vehicle
            l, r, t, b = -veh_width / 2, veh_width / 2, veh_height / 2, -veh_height / 2
            vehicle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            vehicle.set_color(0, 0, 255)

            x_ini, y_ini, theta_ini = self.state
            tras_x = img_x_orig + y_ini*m2pixel
            tras_y = img_y_orig + veh_height/2 + x_ini*m2pixel
            self.veh_transform = rendering.Transform(translation=(tras_x, tras_y), rotation=-theta_ini*np.pi/180)
            vehicle.add_attr(self.veh_transform)
            self.viewer.add_geom(vehicle)

            # add road lanes
            left_lane = rendering.Line((img_x_orig - lane_width / 2, 0), (img_x_orig - lane_width / 2, screen_height))
            left_lane.set_color(0, 0, 0)
            self.viewer.add_geom(left_lane)
            right_lane = rendering.Line((img_x_orig + lane_width / 2, 0), (img_x_orig + lane_width / 2, screen_height))
            right_lane.set_color(0, 0, 0)
            self.viewer.add_geom(right_lane)
    
        # render current vehicle position
        x, y, theta = self.state
        tras_x = img_x_orig + y*m2pixel
        tras_y = img_y_orig + veh_height/2 + x*m2pixel
        self.veh_transform.set_translation(tras_x, tras_y)
        self.veh_transform.set_rotation(-theta*np.pi/180)
        time.sleep(0.1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _is_done(self):
        x,y,_ = self.state
        goal_threshold = 1
        if x >= (self.x_goal - goal_threshold) and x <= (self.x_goal + goal_threshold) and y >= (self.y_goal - goal_threshold) and y <= (self.y_goal + goal_threshold):
            done = True
        else:
            done = False
        return done
    
    def close(self):
        return self.render(close=True)
    
    def _reward(self):
        _,y,_ = self.state
        if y >= self.right_lane or y <= self.left_lane:
            reward = -100
        else:
            reward = 1
        return reward