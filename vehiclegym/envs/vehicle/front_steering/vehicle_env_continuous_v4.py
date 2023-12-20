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
    def __init__(self, x_goal, y_goal, circuit_number, obs = False):
        super(VehicleTfmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Maximum steering position
        self.max_steering = 20
        
        # Action space (normalized between -1 and 1)
        high = np.array([1, 1], dtype = np.float32)
        self.action_space = spaces.Box(-high, high)
        
        # Observation space (normalized between -1 and 1)
        high = np.array([1, 1, 1, 1], dtype = np.float32) #[1, 1, 1]
        low = np.array([-1, -1, 0, 0], dtype = np.float32) #[-1, -1, 0]
        self.observation_space = spaces.Box(low, high)
        
        # X and Y coordinates of the destination
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.destination_reached = False
        
        # Initial position
        self.x_initial = 0
        self.y_initial = -0.2
        
        # Circuit number
        self.circuit_number = circuit_number
        self.obs = obs
        
        # Load vehicle model
        model_path = '../../02_Libraries/Chasis.fmu'
        self.model = load_fmu(model_path)
        
        # Vehicle look ahead distance
        self.lad_obs = 4#6
        
        # Load circuit specifications
        if self.circuit_number == 1:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit1.csv')
            self.lane_width = 1
            self.x_obs = [10]
            self.y_obs = [0]
            self.obs_vel = [2]
            self.obs_radius = 0.35
            self.obs_transform = []
            self.obs_pos = (0,0)
        elif self.circuit_number == 2:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit2.csv')
            self.lane_width = 2
            self.x_obs = [10]
            self.y_obs = [0]
            self.obs_vel = [2]
            self.obs_radius = 0.35
            self.obs_transform = []
            self.obs_pos = (0,0)
        else:
            self.circuit = None
            self.lane_width = 1
        
        # Center lane of the selected circuit
        self.lane_center = [np.array(self.circuit['chasis.x']),\
                            np.array(self.circuit['chasis.y']),\
                            np.array(self.circuit['chasis.theta_out'])]

        # Simulation sample time
        self.sample_time = 0.01
        # Initialization of simulation start and stop times
        self.start_time = 0
        self.stop_time = self.sample_time
        
        # Variable initialization
        self.done = False
        self.sensors = (0,0,0,0)
        self.state = self.reset(self.x_obs, self.y_obs, self.obs_vel)
        self.action_old = 0
        self.viewer = None
        self.display = None
        self.veh_transform = None
        
        # Reward initialization variables
        self.action_old = 0

    def step(self, action, iteration):
        # Execute one time step within the environment
        
        # Convert steering from normalized value to real value
        steering = action[0][0]*self.max_steering
        steering = [np.array(steering)]
        self.model.set(list(['delta']), list(steering))
        pedal = action[0][1]
        pedal = [np.array(pedal)]
        self.model.set(list(['Pedal']), list(pedal))
        # self.model.set(list(['delta']), list([np.array(0)]))
        
        # Simulation options
        opts = self.model.simulate_options()
        opts['ncp'] = 10
        opts['initialize'] = False
        
        # Run the simulation one sample time
        res = self.model.simulate(start_time=self.start_time, final_time=self.stop_time, options=opts)
        
        # Get sensor measurements from the vehicle model
        res_sensors = tuple([res.final(k) for k in ['x','y','theta_out', 'vel']])
        self.sensors = (res_sensors[0] + self.x_initial, res_sensors[1] + self.y_initial, res_sensors[2], res_sensors[3])
        a_lat = np.tan(steering[0]*np.pi/180)*res_sensors[3]*res_sensors[3]/(0.12 + 0.16)
        info = (res_sensors[0] + self.x_initial, res_sensors[1] + self.y_initial, res_sensors[2], a_lat, res_sensors[3])
        
        # print("Action ->  {}".format(action))
        # print("Delta ->  {}".format(res.final('delta')))
        
        # Get the state forwarded to the agent
        self.state = self._lateralcalc()
        
        # Check if environment is terminated
        self.done = self._is_done()
        
        # If environment is not terminated, increase simulation start and stop times one sample time
        if not self.done:
            self.start_time = self.stop_time
            self.stop_time += self.sample_time

        return self.state, self._reward(steering, iteration), self.done, info
     
    def reset(self, x_obs, y_obs, vel_obs):
        # Reset the state of the environment to an initial state
        
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.obs_vel = vel_obs
        self.obs_pos = (x_obs, y_obs)
        
        # Model reset
        self.model.reset()
        self.destination_reached = False
        
        # Reset the model to initial state
        res = self.model.simulate(start_time=0, final_time=0)
        
        # Get sensor measurements from the vehicle model
        res_sensors = tuple([res.final(k) for k in ['x','y','theta_out', 'vel']])
        self.sensors = (res_sensors[0] + self.x_initial, res_sensors[1] + self.y_initial, res_sensors[2], res_sensors[3])
        
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
        obs_radius = self.obs_radius*m2pixel
        
        # Render circuit
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            
            # Add vehicle
            l, r, t, b = -veh_height/2, veh_height/2, veh_width/2, -veh_width/2
            vehicle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            vehicle.set_color(0, 0, 255)
            
            # Vehicle initial position in the screen
            x_ini, y_ini, theta_ini, _ = self.sensors
            tras_x = x_track_ini + x_ini*m2pixel
            tras_y = y_track_ini + y_ini*m2pixel
            self.veh_transform = rendering.Transform(translation=(tras_x, tras_y), rotation=theta_ini*np.pi/180)
            vehicle.add_attr(self.veh_transform)
            self.viewer.add_geom(vehicle)
            
            # Add obstacle
            if self.obs == True:
                if self.circuit_number == 2:
                    for x_obs, y_obs in zip(self.x_obs, self.y_obs):
                        obstacle = rendering.make_circle(obs_radius)
                        obstacle.set_color(0, 0, 0)
                        x_obs = x_track_ini + x_obs*m2pixel
                        y_obs = y_track_ini + y_obs*m2pixel
                        obs_transform = rendering.Transform(translation=(x_obs, y_obs), rotation=0)
                        self.obs_transform.append(obs_transform)
                        obstacle.add_attr(obs_transform)
                        self.viewer.add_geom(obstacle)

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
        x, y, theta, _ = self.sensors
        tras_x = x_track_ini + x*m2pixel
        tras_y = y_track_ini + y*m2pixel
        self.veh_transform.set_translation(tras_x, tras_y)
        self.veh_transform.set_rotation(theta*np.pi/180)
        
        # Render obstacle
        if self.obs == True:
            i = 0
            x_obs_list = []
            y_obs_list = []
            for x_obs, y_obs, vel_obs in zip(self.obs_pos[0],self.obs_pos[1], self.obs_vel):
                x_obs = min(x_obs + vel_obs*self.sample_time, 25)
                # y_obs = max(y_obs - vel_obs*self.sample_time, -1)
                x_obs_list.append(x_obs)
                y_obs_list.append(y_obs)
                x_obs = x_track_ini + x_obs*m2pixel
                y_obs = y_track_ini + y_obs*m2pixel
                self.obs_transform[i].set_translation(x_obs, y_obs)
                i += 1
            self.obs_pos = (x_obs_list, y_obs_list)
        
        # time.sleep(0.1)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _is_done(self):
        # Check if environment is terminated
        
        # Get state
        # e_lat, e_theta, _, _, _ = self.state
        e_lat, e_theta, _, _ = self.state
        # Vehicle sensors
        x, y, _, vel = self.sensors
        
        # Distance to obstacle        
        dist_obs = np.sqrt((self.obs_pos[0]-x)**2 + (self.obs_pos[1]-y)**2)
        
        # Set maximum simulation time depending on the selected circuit
        if self.circuit_number == 1:
            done_time = 300
        else:
            done_time = 100
        
        if abs(e_lat) > 1:
            # Vehicle exceeded road boundaries
            done = True
        elif self.stop_time >= done_time:
            # Simulation exceeded maximum simulation time
            done = True
        elif self.circuit_number == 2 and x >= self.x_goal:
            # Vehicle reached final destination
            done = True
            self.destination_reached = True
        elif self.circuit_number == 1 and x <= (self.x_goal + 0.2) and x >= (self.x_goal - 0.2) and y <= (self.y_goal + 0.2) and y >= (self.y_goal - 0.2):
            # Vehicle reached final destination
            done = True
            self.destination_reached = True
        elif self.circuit_number == 2 and x < -0.01:
            # Vehicle out of circuit
            done = True
        elif self.obs == True and any(dist <= self.obs_radius for dist in dist_obs):
            done = True
        elif vel < 0:
            done = True
        else:
            done = False
        
        return done
    
    def close(self):
        return self.render(close=True)
    
    def _reward(self, action, iteration):
        # Calculate reward value
        
        # Get sensors
        x,y,theta,vel = self.sensors
        
        # Get state
        # e_lat, _, vel, x_obser, overtake = self.state
        e_lat, _, _, _ = self.state

        # Reward due to lateral acceleration
        a_lat = np.tan(action[0]*np.pi/180)*vel*vel/(0.12 + 0.16)
        if abs(a_lat) >= 8:
            reward_ay = 0#(8-abs(a_lat))/32*5
        else:
            reward_ay = 0
        
        # Reward due to vehicle exceeding road boundaries
        # Reward out of boundaries only applied if vehicle exceedes road boundaries
        if abs(e_lat) > 0.9:
            reward_out_boundaries = (0.9 - abs(e_lat))/0.1*10
        else:
            reward_out_boundaries = 0
            
        # Reward obstacles
        if len(self.obs_pos[0]) ==  1:
            x_obs = self.obs_pos[0][0]
            y_obs = self.obs_pos[1][0]
        else:
            for i in range(len(self.obs_pos[0])):
                if x > self.obs_pos[0][i] and i != len(self.obs_pos[0])-1:
                    continue
                else:
                    x_obs = self.obs_pos[0][i]
                    y_obs = self.obs_pos[1][i]
                    break
        dist_obs = np.sqrt((x_obs - x)**2 + (y_obs - y)**2)
        
        dist_obs = np.sqrt((self.obs_pos[0]-x)**2 + (self.obs_pos[1]-y)**2)
        dist_obs = min(dist_obs)
        
        if self.obs == True and (dist_obs - 0.7) <= 0:
            reward_obs = (dist_obs - 0.7)/0.4*10
        else:
            reward_obs = 0
        
        # Reward if vehicle follows the center lane of the road
        lane_std = 0.2
        reward_lane = np.exp(-e_lat**2/(2*lane_std**2))
        reward_lane = (-abs(e_lat)+1)*2
        reward_lane = reward_lane + x/self.x_goal*5
        # if x_obser == 0:
        #     reward_lane = 1*reward_lane*vel
        # else:
        #     reward_lane = 1*reward_lane
        # if overtake == 0 and x_obser == 0:
        #     reward_lane = 1*reward_lane*vel
        # elif overtake == 0 and x_obser != 0:
        #     reward_lane = 1*reward_lane+(1-abs(vel*6-self.obs_vel[0]))
        # elif overtake == 1:
        #     reward_lane = 1*reward_lane*vel
        
        # Reward if destination reached
        if self.destination_reached == True:
            reward_destination = 10000#x/self.x_goal*5
        else:
            reward_destination = 0#x/self.x_goal*5
        
        # print("reward_obs ->  {}".format(reward_obs))
        
        if vel <= 0.7:
            reward_vel = (vel - 0.7)/0.7*10
        else:
            reward_vel = 0
        
        return reward_lane + reward_out_boundaries + reward_obs + reward_destination + reward_ay + reward_vel
    
    def _lateralcalc(self):
        # Function to calculate the lateral distance from the vehicle to the center lane
        # The lateral distance is the minumum distance from the vehicle to the center lane
        
        # Get vehicle sensor information
        x,y,theta,vel = self.sensors
        
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
        
        # convert vehicle heading frame to trajectory heading frame
        if theta > 180:
            theta = theta - 360
        
        # calculate etheta
        etheta = theta - pt2_pt1_ag*180/np.pi
        
        # etheta adaptation
        if etheta >= 180:
            etheta = etheta - 360
        elif etheta <= -180:
            etheta = etheta + 360
        etheta = -etheta/180
        
        # Normalize velocity between 0 and 1
        vel = vel/6
        
        # Longitudinal distance to obstacle
        if self.obs == True:
            if len(self.obs_pos[0]) ==  1:
                x_obs = self.obs_pos[0][0]
                y_obs = self.obs_pos[1][0]
            else:
                for i in range(len(self.obs_pos[0])):
                    if x > self.obs_pos[0][i] and i != len(self.obs_pos[0])-1:
                        continue
                    else:
                        x_obs = self.obs_pos[0][i]
                        y_obs = self.obs_pos[1][i]
                        break
            x_obser = x_obs - x
            x_obser = min(x_obser, self.lad_obs)
            x_obser = max(x_obser, -self.lad_obs)
            
            # Normalize longitudinal distance to obstacle between 0 and 1
            x_obser = np.clip(x_obser/self.lad_obs, 0, 1)
            x_obser = 1 - x_obser
            if x >= x_obs:
                x_obser = 0
            
            # Obstacle lateral distance
            y_obser = elat + 2*y_obs/self.lane_width
        else:
            x_obser = 0
            y_obser = 0
        
        if x > 10.5:
            overtake = 1
        else:
            overtake = 0
        
        return elat, etheta, vel, x_obser#, overtake