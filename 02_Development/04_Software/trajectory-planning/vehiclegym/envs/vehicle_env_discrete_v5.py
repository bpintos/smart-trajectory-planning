# -*- coding: utf-8 -*-
"""
@author: Borja Pintos
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd

class VehicleTfmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, starting_pose, destination_pose, circuit_number, obs, obs_coor, sensor_discretization, actions, states):
        super(VehicleTfmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Action space
        self.action_space = spaces.Discrete(len(actions))
        
        # Observation space
        self.observation_space = spaces.Tuple((
        spaces.Discrete(len(states[0])),
        spaces.Discrete(len(states[1])),
        spaces.Discrete(len(states[2]))
        ))
        self.states = states
        self.sensor_discretization = sensor_discretization
        
        # X and Y coordinates of the final destination
        self.destination_pose = destination_pose
        
        # Initial position
        self.x_initial = starting_pose[0]
        self.y_initial = starting_pose[1]
        self.heading_initial = starting_pose[2]
        
        # Vehicle velocity
        self.velocity = 0.5 # Not used
        
        # Circuit number
        self.circuit_number = circuit_number
        self.obs = obs
        if obs:
            self.obstacles = obs_coor
        else:
            self.obstacles = []
        
        # Load circuit specifications
        if self.circuit_number == 1:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit1.csv')
            self.lane_width = 1
            self.x_obs = 10
            self.y_obs = 0
            self.obs_radius = 0.1
            self.obs_transform = []
        elif self.circuit_number == 2:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit2.csv')
            self.lane_width = 4
            self.x_obs = 10
            self.y_obs = 0
            self.obs_radius = 0.1
            self.obs_transform = []
        else:
            self.circuit = None
            self.lane_width = 1

        # Simulation sample time and simulation start and stop times
        self.sample_time = 1
        self.start_time = 0
        self.stop_time = self.sample_time
        
        # Variable initialization
        self.done = False
        self.state = self.reset(0)
        self.viewer = None
        self.display = None
        self.veh_transform = None

    def reset(self, ep):
        # Reset the state of the environment to an initial state
        
        if ep < 1000:
            self.x_initial = 16
            if ep%5 == 0:
                self.y_initial = 0
            elif ep%5 == 1:
                self.y_initial = -1
            elif ep%5 == 2:
                self.y_initial = 1
            elif ep%5 == 3:
                self.y_initial = 0
            elif ep%5 == 4:
                self.y_initial = 0
        elif ep < 2000:
            self.x_initial = 14
            if ep%5 == 0:
                self.y_initial = 0
            elif ep%5 == 1:
                self.y_initial = -1
            elif ep%5 == 2:
                self.y_initial = 1
            elif ep%5 == 3:
                self.y_initial = 0
            elif ep%5 == 4:
                self.y_initial = 0
        elif ep < 5000:
            self.x_initial = 4
            if ep%5 == 0:
                self.y_initial = 0
            elif ep%5 == 1:
                self.y_initial = -1
            elif ep%5 == 2:
                self.y_initial = 1
            elif ep%5 == 3:
                self.y_initial = 0
            elif ep%5 == 4:
                self.y_initial = 0
        else:
            self.x_initial = 0
            if ep%5 == 0:
                self.y_initial = 0
            elif ep%5 == 1:
                self.y_initial = -1
            elif ep%5 == 2:
                self.y_initial = 1
            elif ep%5 == 3:
                self.y_initial = 0
            elif ep%5 == 4:
                self.y_initial = 0
        
        # Reset the model to initial state
        self.sensors = (self.x_initial, self.y_initial, 0)
        
        # Get the state forwarded to the agent
        self.state = self._get_state()
        
        # Reset simulation start and stop times
        self.start_time = 0
        self.stop_time = self.sample_time
        
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        
        # Action = 0 -> Move up
        # Action = 1 -> Keep lateral position
        # Action = 2 -> Move down
        if action[0] == 0:          
            sens = list(self.sensors)
            sens[1] = round(sens[1] - self.sensor_discretization[1],1)
            sens = tuple(sens)
            self.sensors = sens
        elif action[0] == 1:
            sens = list(self.sensors)
            sens[0] = round(sens[0] + self.sensor_discretization[0],1)
            sens = tuple(sens)
            self.sensors = sens
        elif action[0] == 2:
            sens = list(self.sensors)
            sens[1] = round(sens[1] + self.sensor_discretization[1],1)
            sens = tuple(sens)
            self.sensors = sens
        
        # Get the state forwarded to the agent
        self.state = self._get_state()
        
        # Check if environment is terminated
        self.done = self._is_done()
        
        # If environment is not terminated, increase simulation start and stop times one sample time
        if not self.done:
            self.start_time = self.stop_time
            self.stop_time += self.sample_time

        return self.state, self._reward(action), self.done, self.sensors
    
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
        lane_center_x = np.array(self.circuit['chasis.x'])
        # Y coordinate of the circuit's center lane
        lane_center_y = np.array(self.circuit['chasis.y'])
        # Heading of the circuit's center lane
        lane_center_theta = np.array(self.circuit['chasis.theta_out'])
        
        # Convertion from meter to pixel
        lane_width = self.lane_width*m2pixel
        obs_radius = self.obs_radius*m2pixel
        
        # Render circuit
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            
            # Add vehicle
            
            # Convertion from meter to pixel
            veh_width = veh_width*m2pixel
            veh_height = veh_height*m2pixel
            l, r, t, b = -veh_height/2, veh_height/2, veh_width/2, -veh_width/2
            vehicle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            vehicle.set_color(0, 0, 255)
            
            # Vehicle initial position in the screen
            x_ini, y_ini, theta_ini = self.sensors
            
            # Convertion from meter to pixel
            tras_x = x_track_ini + x_ini*m2pixel
            tras_y = y_track_ini + y_ini*m2pixel
            self.veh_transform = rendering.Transform(translation=(tras_x, tras_y), rotation=theta_ini*np.pi/180)
            vehicle.add_attr(self.veh_transform)
            self.viewer.add_geom(vehicle)
            
            # Add obstacle
            if self.obs == True:
                if self.circuit_number == 2:
                    for obstacle in self.obstacles:
                        obs = rendering.make_circle(obs_radius)
                        obs.set_color(0, 0, 0)
                        x_obs = x_track_ini + obstacle[0]*m2pixel
                        y_obs = y_track_ini + obstacle[1]*m2pixel
                        obs_transform = rendering.Transform(translation=(x_obs, y_obs), rotation=0)
                        self.obs_transform.append(obs_transform)
                        obs.add_attr(obs_transform)
                        self.viewer.add_geom(obs)

            # Add left and right road lanes
            x_left_rel, y_left_rel = 0, self.lane_width/2
            x_right_rel, y_right_rel = 0, -self.lane_width/2
            
            theta = lane_center_theta[0]*np.pi/180
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            
            x_left, y_left = np.matmul(R,[x_left_rel, y_left_rel])
            x_left_prev, y_left_prev = x_left, y_left
            x_right, y_right = np.matmul(R,[x_right_rel, y_right_rel])
            x_right_prev, y_right_prev = x_right, y_right
            
            for x,y,theta in zip(lane_center_x, lane_center_y, lane_center_theta):
                
                # Add middle lane
                x_prev, y_prev = x_track_ini, y_track_ini
                x_center = x*m2pixel + x_track_ini
                y_center = y*m2pixel + y_track_ini
                center_lane = rendering.Line((x_prev, y_prev), (x_center, y_center))
                center_lane.set_color(0, 0, 0)
                self.viewer.add_geom(center_lane)
                x_prev, y_prev = x_center, y_center
                
                theta = theta*np.pi/180
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                
                # Add left lane
                x_left, y_left = np.matmul(R,[x_left_rel, y_left_rel])
                x_left = x + x_left
                y_left = y + y_left
                # Convert to pixel
                x_left_prev_pix = x_left_prev*m2pixel + x_track_ini
                y_left_prev_pix = y_left_prev*m2pixel + y_track_ini
                x_left_pix = x_left*m2pixel + x_track_ini
                y_left_pix = y_left*m2pixel + y_track_ini
                left_lane = rendering.Line((x_left_prev_pix, y_left_prev_pix), (x_left_pix, y_left_pix))
                left_lane.set_color(0, 255, 0)
                self.viewer.add_geom(left_lane)
                x_left_prev, y_left_prev = x_left, y_left
                
                # Add right lane
                x_right, y_right = np.matmul(R,[x_right_rel, y_right_rel])
                x_right = x + x_right
                y_right = y + y_right
                # Convert to pixel
                x_right_prev_pix = x_right_prev*m2pixel + x_track_ini
                y_right_prev_pix = y_right_prev*m2pixel + y_track_ini
                x_right_pix = x_right*m2pixel + x_track_ini
                y_right_pix = y_right*m2pixel + y_track_ini
                right_lane = rendering.Line((x_right_prev_pix, y_right_prev_pix), (x_right_pix, y_right_pix))
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
        
        # Vehicle sensors
        x, y, hea = self.sensors
        
        # Set maximum simulation time depending on the selected circuit
        if self.circuit_number == 1:
            done_time = 500
        else:
            done_time = 500
        
        if y <= -2 or y >= 2 or x >= 20:
            # Vehicle exceeded road boundaries
            done = True
        elif self.stop_time >= done_time:
            # Simulation exceeded maximum simulation time
            done = True
        elif [x,y,hea] == self.destination_pose:
            # Vehicle reached final destination
            done = True
        elif self.circuit_number == 2 and x < 0:
            # Vehicle out of circuit
            done = True
        elif (x, y) in self.obstacles:
            done = True
        else:
            done = False
        
        return done
    
    def close(self):
        return self.render(close=True)
    
    def _reward(self, action):
        # Calculate reward value
             
        # Get sensors
        x, y, hea = self.sensors
        
        # Reward out of boundaries only applied if vehicle exceedes road boundaries
        if y <= -2 or y >= 2 or x >= 20:
            reward_out_boundaries = -1
        else:
            reward_out_boundaries = 0
            
        # Reward obstacles
        if (x, y) in self.obstacles:
            reward_obs = -1
        else:
            reward_obs = 0
            
        # Reward if destination reached
        if [x,y,hea] == self.destination_pose:
            reward_destination = 1
        else:
            reward_destination = 0
        
        return reward_out_boundaries + reward_obs + reward_destination
    
    def _get_state(self):
        x, y, hea = self.sensors
        laser_1 = self.states[0][-1]
        laser_2 = self.states[1][-1]
        laser_3 = self.states[2][-1]
        # Laser 1
        for i,j in zip(self.states[0],range(0, len(self.states[0]))):
            if (round(x+i,1),y) in self.obstacles:
                laser_1 = self.states[0][j]
                break
        # Laser 2
        for i,j in zip(self.states[1],range(0, len(self.states[1]))):
            if (x,round(y+i,1)) in self.obstacles:
                laser_2 = self.states[1][j]
                break
            if (x,round(y+i,1)) in ((x,2),(x,-2)):
                laser_2 = self.states[1][j]
                break
        # Laser 3
        for i,j in zip(self.states[2],range(0, len(self.states[2]))):
            if (x,round(y-i,1)) in self.obstacles:
                laser_3 = self.states[2][j]
                break
            if (x,round(y-i,1)) in ((x,2),(x,-2)):
                laser_3 = self.states[1][j]
                break
        return (laser_1,laser_2,laser_3)
        