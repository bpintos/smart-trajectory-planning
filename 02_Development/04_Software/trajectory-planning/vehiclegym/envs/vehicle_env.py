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
import pandas as pd

class VehicleTfmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, x_goal, y_goal, circuit_number):
        super(VehicleTfmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Maximum steering position
        self.max_steering = 20
        
        # Action space (normalized between -1 and 1)
        high = np.array([1], dtype = np.float32)
        self.action_space = spaces.Box(-high, high)
        
        # Observation space (normalized between -1 and 1)
        high = np.array([1, 1], dtype = np.float32)
        low = np.array([-1, -1], dtype = np.float32)
        self.observation_space = spaces.Box(low, high)
        
        # X and Y coordinates of the destination
        self.x_goal = x_goal
        self.y_goal = y_goal
        
        # Initial position
        self.x_initial = 0
        self.y_initial = 0
        
        # Circuit number
        self.circuit_number = circuit_number
        
        # Load vehicle model
        model_path = '../../02_Libraries/Chasis_simpl.fmu'
        self.model = load_fmu(model_path)
        
        # Load circuit specifications
        if self.circuit_number == 1:
            self.circuit = pd.read_csv('vehiclegym/envs/circuit1.csv')
        elif self.circuit_number == 2:
            self.circuit = pd.read_csv('vehiclegym/envs/circuit2.csv')
        else:
            self.circuit = None
        # Center lane of the selected circuit
        self.lane_center = [np.array(self.circuit['chasis.x']),\
                            np.array(self.circuit['chasis.y']),\
                            np.array(self.circuit['chasis.theta_out'])]
        # Circuit's lane width
        self.lane_width = 1
        
        # Simulation sample time
        self.sample_time = 0.01
        # Initialization of simulation start and stop times
        self.start_time = 0
        self.stop_time = self.sample_time
        
        # Variable initialization
        self.done = False
        self.sensors = (0,0,0)
        self.state = self.reset(0)
        self.action_old = 0
        self.viewer = None
        self.display = None
        self.veh_transform = None
        
        # Reward initialization variables
        self.action_old = 0

    def step(self, action, iteration, ep):
        # Execute one time step within the environment
        
        # Convert steering from normalized value to real value
        action = action[0]*self.max_steering
        action = [np.array(action)]
        self.model.set(list(['delta']), list(action))
        
        # Simulation options
        opts = self.model.simulate_options()
        opts['ncp'] = 10
        opts['initialize'] = False
        
        # Run the simulation one sample time
        res = self.model.simulate(start_time=self.start_time, final_time=self.stop_time, options=opts)
        
        # Get sensor measurements from the vehicle model
        res_sensors = tuple([res.final(k) for k in ['x','y','theta_out']])
        if ep >= 0:
            self.y_initial = 0.2
        self.sensors = (res_sensors[0] + self.x_initial, res_sensors[1] + self.y_initial, res_sensors[2])
        
        print("Action ->  {}".format(action))
        print("Delta ->  {}".format(res.final('delta')))
        
        # Get the state forwarded to the agent
        self.state = self._lateralcalc()
        
        # Check if environment is terminated
        self.done = self._is_done()
        
        # If environment is not terminated, increase simulation start and stop times one sample time
        if not self.done:
            self.start_time = self.stop_time
            self.stop_time += self.sample_time

        return self.state, self._reward(action, iteration), self.done, {}
     
    def reset(self, ep):
        # Reset the state of the environment to an initial state
        
        # Model reset
        self.model.reset()
        
        # Reset the model to initial state
        res = self.model.simulate(start_time=0, final_time=0)
        
        # Get sensor measurements from the vehicle model
        res_sensors = tuple([res.final(k) for k in ['x','y','theta_out']])
        if ep >= 0:
            self.y_initial = 0.2
        self.sensors = (res_sensors[0] + self.x_initial, res_sensors[1] + self.y_initial, res_sensors[2])
        
        # Get the state forwarded to the agent
        self.state = self._lateralcalc()
        
        # Reset simulation start and stop times
        self.start_time = 0
        self.stop_time = self.sample_time
        
        return self.state
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        
        # If close is true, close the render screen
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return True
        
        # Load render parameters depending on the selected circuit
        if self.circuit_number == 1:
            # Screen size
            screen_width = 1200
            screen_height = 900      
            
            # Vehicle initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 30
            
            # Vehicle size
            veh_width = 0.2
            veh_height = 0.4
            
            # Circuit lane's width
            self.lane_width = 1
            
        elif self.circuit_number == 2:
            # Screen size
            screen_width = 2000
            screen_height = 900      
            
            # Vehicle initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 60
            
            # Vehicle size
            veh_width = 0.2
            veh_height = 0.4
            
            # Circuit lane's width
            self.lane_width = 1
            
        else:
            # Screen size
            screen_width = 1200
            screen_height = 900      
            
            # Vehicle initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 30
            
            # Vehicle size
            veh_width = 0.2
            veh_height = 0.4
            
            # Circuit lane's width
            self.lane_width = 1
        
        # X coordinate of the circuit's center lane
        lane_center_x = self.lane_center[0]
        # Y coordinate of the circuit's center lane
        lane_center_y = self.lane_center[1]
        # Heading of the circuit's center lane
        lane_center_theta = self.lane_center[2]
        
        # Convertion from meter to pixel
        veh_width = veh_width*m2pixel
        veh_height = veh_height*m2pixel
        lane_width = self.lane_width*m2pixel
        
        # Render circuit
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            
            # Add vehicle
            l, r, t, b = -veh_height / 2, veh_height / 2, veh_width / 2, -veh_width / 2
            vehicle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            vehicle.set_color(0, 0, 255)
            
            # Vehicle initial position in the screen
            x_ini, y_ini, theta_ini = self.sensors
            tras_x = x_track_ini + x_ini*m2pixel
            tras_y = y_track_ini + y_ini*m2pixel
            self.veh_transform = rendering.Transform(translation=(tras_x, tras_y), rotation=theta_ini*np.pi/180)
            vehicle.add_attr(self.veh_transform)
            self.viewer.add_geom(vehicle)

            # Add left and right road lanes
            x_left_rel, y_left_rel = 0, lane_width/2
            x_right_rel, y_right_rel = 0, -lane_width/2
            x_prev, y_prev = x_track_ini, y_track_ini
            theta = lane_center_theta[0]*np.pi/180
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            x_left, y_left = np.matmul(R,[x_left_rel, y_left_rel])
            x_left_prev, y_left_prev = x_left + x_track_ini, y_left + y_track_ini
            x_right, y_right = np.matmul(R,[x_right_rel, y_right_rel])
            x_right_prev, y_right_prev = x_right + x_track_ini, y_right + y_track_ini
            for x,y,theta in zip(lane_center_x, lane_center_y, lane_center_theta):
                x_center = x*m2pixel + x_track_ini
                y_center = y*m2pixel + y_track_ini
                center_lane = rendering.Line((x_prev, y_prev), (x_center, y_center))
                center_lane.set_color(0, 0, 0)
                self.viewer.add_geom(center_lane)
                x_prev, y_prev = x_center, y_center
                
                theta = theta*np.pi/180
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                
                x_left, y_left = np.matmul(R,[x_left_rel, y_left_rel])
                x_left = x*m2pixel + x_left + x_track_ini
                y_left = y*m2pixel + y_left + y_track_ini
                left_lane = rendering.Line((x_left_prev, y_left_prev), (x_left, y_left))
                left_lane.set_color(0, 255, 0)
                self.viewer.add_geom(left_lane)
                x_left_prev, y_left_prev = x_left, y_left
                
                x_right, y_right = np.matmul(R,[x_right_rel, y_right_rel])
                x_right = x*m2pixel + x_right + x_track_ini
                y_right = y*m2pixel + y_right + y_track_ini
                right_lane = rendering.Line((x_right_prev, y_right_prev), (x_right, y_right))
                right_lane.set_color(255, 0, 0)
                self.viewer.add_geom(right_lane)
                x_right_prev, y_right_prev = x_right, y_right
    
        # Render current vehicle position
        x, y, theta = self.sensors
        tras_x = x_track_ini + x*m2pixel
        tras_y = y_track_ini + y*m2pixel
        self.veh_transform.set_translation(tras_x, tras_y)
        self.veh_transform.set_rotation(theta*np.pi/180)
        
        # time.sleep(0.1)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _is_done(self):
        # Check if environment is terminated
        
        # Area of the final destination
        area_destination = 0.5
        x_dest_ini, x_dest_fin = self.x_goal - area_destination, self.x_goal + area_destination
        y_dest_ini, y_dest_fin = self.y_goal - area_destination, self.y_goal + area_destination
        
        # Get state
        e_lat, _ = self.state
        # Vehicle sensors
        x, y, _ = self.sensors
        
        # Set maximum simulation time depending on the selected circuit
        if self.circuit_number == 1:
            done_time = 30
        else:
            done_time = 10
        
        if abs(e_lat) > 1:
            # Vehicle exceeded road boundaries
            done = True
        elif self.stop_time >= done_time:
            # Simulation exceeded maximum simulation time
            done = True
        elif x >= x_dest_ini and x <= x_dest_fin and y >= y_dest_ini and y <= y_dest_fin:
            # Vehicle reached final destination
            done = True
        else:
            done = False
        return done
    
    def close(self):
        return self.render(close=True)
    
    def _reward(self, action, iteration):
        # Calculate reward value
        
        # Reward steering gradient
        # Steering gradient calculation
        if iteration == 0:
            self.action_old = action
        action_grad_max = 10
        action = action[0]
        action_grad = (action - self.action_old)/self.sample_time
        
        # Reward gradient only applied if steering gradient exceedes maximum limit
        if abs(action_grad) > action_grad_max:
            reward_action_grad = max(-0.01*abs(action_grad)/action_grad_max,-0.1)
        else:
            reward_action_grad = 0
                    
        self.action_old = action
        
        # Reward due to vehicle exceeding road boundaries
        # Get state
        e_lat, _ = self.state
        
        # Reward out of boundaries only applied if vehicle exceedes road boundaries
        if abs(e_lat) > 1:
            reward_out_boundaries = -100
        else:
            reward_out_boundaries = 0
        
        # Reward if vehicle follows the center lane of the road
        lane_std = 0.1
        reward_lane = np.exp(-e_lat**2/(2*lane_std**2))
        
        return reward_lane + reward_out_boundaries + reward_action_grad
    
    def _lateralcalc(self):
        # Function to calculate the lateral distance from the vehicle to the center lane
        # The lateral distance is the minumum distance from the vehicle to the center lane
        
        # Get vehicle sensor information
        x,y,theta = self.sensors
        
        # Get coordinates of center lane
        lane_center_x = self.lane_center[0]
        lane_center_y = self.lane_center[1]
        
        # Calculate distance from vehicle position to each point of the center lane
        dist = np.sqrt((x-lane_center_x)**2+(y-lane_center_y)**2)
        # Sort the distances
        distarg = np.argsort(dist)
        # Take the index from the minimum distance
        pt1_idx = np.min((distarg[0], distarg[1]))
        pt2_idx = np.max((distarg[0], distarg[1]))
        
        # More accurate calculation of the lateral distance
        pt1 = [lane_center_x[pt1_idx], lane_center_y[pt1_idx]]
        pt2 = [lane_center_x[pt2_idx], lane_center_y[pt2_idx]]
        pt2_pt1 = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
        pt2_pt1_ag = np.arctan2(pt2_pt1[1], pt2_pt1[0])
        pos_pt1 = [x - pt1[0], y - pt1[1]]
        pos_pt1_proj = (pt2_pt1[0]*pos_pt1[0] + pt2_pt1[1]*pos_pt1[1])/np.sqrt(pt2_pt1[0]**2+pt2_pt1[1]**2)
        ptelat = [pt1[0] + pos_pt1_proj*np.cos(pt2_pt1_ag), 
                   pt1[1] + pos_pt1_proj*np.sin(pt2_pt1_ag)]
        pos_ptelat = [x - ptelat[0], y - ptelat[1]]
        c, s = np.cos(-pt2_pt1_ag), np.sin(-pt2_pt1_ag)
        R = np.array(((c, -s), (s, c)))
        _, elat = np.matmul(R,pos_ptelat)
        
        # Normalization between -1 and 1 of the lateral distance
        elat = -2*elat/self.lane_width
        
        # etheta. This formula only valid for circuit number 2
        if theta >= 180:
            etheta = theta - 360
        else:
            etheta = theta
        etheta = -etheta/180
        
        return elat, etheta