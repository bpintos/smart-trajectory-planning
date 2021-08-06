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
    def __init__(self, x_goal, y_goal, circuit_number, obs = False):
        super(VehicleTfmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Action space
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Tuple((
        spaces.Discrete(5),
        # spaces.Discrete(20),
        spaces.Discrete(4),
        spaces.Discrete(3),
        spaces.Discrete(3)
        ))
        
        # X and Y coordinates of the final destination
        self.x_goal = x_goal
        self.destination_reached = False
        
        # Initial position
        self.x_initial = 0
        self.y_initial = 0
        self.theta_initial = 0
        self.long_dist_old = self.x_initial
        
        # Robot velocity
        self.velocity = 0
        
        # Circuit number
        self.circuit_number = circuit_number
        if obs:
            self.obstacles = [(0,3),(0,10),(1,12),(0,14),(-1,16),(1,18)]
            self.obstacles = [(0,14),(0,15),(1,17),(1,18)]
            self.obstacles = [(0,3),(1,4)]
        else:
            self.obstacles = []
        
        # Load circuit specifications
        if self.circuit_number == 1:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit1.csv')
            self.lane_width = 1
            self.x_obs = 10
            self.y_obs = 0
            self.obs_radius = 0.3
            self.obs_transform = None
        elif self.circuit_number == 2:
            self.circuit = pd.read_csv('vehiclegym/envs/circuits/circuit2.csv')
            self.lane_width = 2
            self.x_obs = 10
            self.y_obs = 0
            self.obs_radius = 0.3
            self.obs_transform = None
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
        self.robot_transform = None

    def step(self, action):
        # Execute one time step within the environment
        
        # Action = 0 -> Only negative rotation. Turn -90 degress anticlockwise
        # Action = 1 -> Move forwards, no rotation
        # Action = 2 -> Only positive rotation. Turn 90 degress clockwise
        # Robot heading = 0 degress matches the global x axis
        if action[0] == 0:
            self.velocity = 0
            sens = list(self.sensors)
            sens[2] = sens[2] - 90
            if sens[2] == -180:
                sens[2] = 180
            sens = tuple(sens)
            self.sensors = sens
        elif action[0] == 1:
            self.velocity = 1
        elif action[0] == 2:
            self.velocity = 0
            sens = list(self.sensors)
            sens[2] = sens[2] + 90
            if sens[2] == 270:
                sens[2] = -90
            sens = tuple(sens)
            self.sensors = sens
        
        # Update x and y coordinates of the robot
        sens = list(self.sensors)
        sens[0] = sens[0] + self.velocity*self.sample_time*np.cos(sens[2]/180*np.pi)
        sens[0] = int(sens[0])
        sens[1] = sens[1] + self.velocity*self.sample_time*np.sin(sens[2]/180*np.pi)
        sens[1] = int(sens[1])
        sens = tuple(sens)
        self.sensors = sens
        
        # Perception system for obstacle detection
        x, y, _ = self.sensors
        lad = 1
        obs_found = False
        for i in range(x+1, x+lad+1):
            if obs_found:
                break
            for j in range (-1,2):
                if (j,i) in self.obstacles:
                    obs_long = 1
                    obs_lat = j
                    obs_found = True
                    break
                else:
                    obs_long = 0
                    obs_lat = 0
        if (y,x) in self.obstacles:
            obs_long = 2
            obs_lat = 0
        
        # State = (lateral_error, longitudinal_distance, robot's heading)
        # For the straigth lane use case, the lateral error matches the y coordinate
        # and the longitudinal distance matches the x coordinate
        self.state = (self.sensors[1], self.sensors[2], obs_long, obs_lat)
        
        # Check if environment is terminated
        self.done = self._is_done()
        
        # If environment is not terminated, increase simulation start and stop times one sample time
        if not self.done:
            self.start_time = self.stop_time
            self.stop_time += self.sample_time

        return self.state, self._reward(action), self.done, self.sensors
     
    def reset(self, ep):
        # Reset the state of the environment to an initial state
        
        self.destination_reached = False
        if ep < 250:
            self.x_initial = 12
            self.long_dist_old = 12
        elif ep < 500:
            self.x_initial = 8
            self.long_dist_old = 8
        elif ep < 750:
            self.x_initial = 4
            self.long_dist_old = 4
        else:
            self.x_initial = 0
            self.long_dist_old = 0
        
        # Reset the model to initial state
        self.sensors = (self.x_initial, self.y_initial, self.theta_initial)
        
        # State = (lateral_error, longitudinal_distance)
        # For the straigth lane use case, the lateral error matches the y coordinate
        # and the longitudinal distance matches the x coordinate
        self.state = (self.sensors[1], self.sensors[2], 0, 0)
        
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
            
            # Robot initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 30
            
            # Robot size
            robot_width = 0.4
            robot_height = 0.4
            
        elif self.circuit_number == 2:
            # Screen size
            screen_width = 2000
            screen_height = 900      
            
            # Robot initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 60
            
            # Robot size
            robot_width = 0.4
            robot_height = 0.4
            
        else:
            # Screen size
            screen_width = 1200
            screen_height = 900      
            
            # Robot initial position in the screen
            x_track_ini, y_track_ini = 350, 400
            
            # Convertion from meters to pixels
            m2pixel = 30
            
            # Robot size
            robot_width = 0.4
            robot_height = 0.4
        
        # X coordinate of the circuit's center lane
        lane_center_x = np.array(self.circuit['chasis.x'])
        # Y coordinate of the circuit's center lane
        lane_center_y = np.array(self.circuit['chasis.y'])
        # Heading of the circuit's center lane
        lane_center_theta = np.array(self.circuit['chasis.theta_out'])
        
        # Convertion from meter to pixel
        robot_width = robot_width*m2pixel
        robot_height = robot_height*m2pixel
        lane_width = self.lane_width*m2pixel
        obs_radius = self.obs_radius*m2pixel
        
        # Render circuit
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            
            # Add robot
            l, r, t, b = -robot_width/2, robot_width/2, robot_height/2, -robot_height/2
            robot = rendering.FilledPolygon([(l, b), (l, t), (r, 0)])
            robot.set_color(0, 0, 1)
            
            # Robot initial position in the screen
            x_ini, y_ini, theta_ini = self.sensors
            tras_x = x_track_ini + x_ini*m2pixel
            tras_y = y_track_ini + y_ini*m2pixel
            self.robot_transform = rendering.Transform(translation=(tras_x, tras_y), rotation=theta_ini*np.pi/180)
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)
            
            # Add obstacle
            # if self.obs == True:
            #     if self.circuit_number == 2:
            #         obstacle = rendering.make_circle(obs_radius)
            #         obstacle.set_color(0, 0, 0)
            #         x_obs = x_track_ini + self.x_obs*m2pixel
            #         y_obs = y_track_ini + self.y_obs*m2pixel
            #         self.obs_transform = rendering.Transform(translation=(x_obs, y_obs), rotation=0)
            #         obstacle.add_attr(self.obs_transform)
            #         self.viewer.add_geom(obstacle)

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
                
                # Left lane
                x_left, y_left = np.matmul(R,[x_left_rel, y_left_rel])
                x_left = x*m2pixel + x_left + x_track_ini
                y_left = y*m2pixel + y_left + y_track_ini
                left_lane = rendering.Line((x_left_prev, y_left_prev), (x_left, y_left))
                left_lane.set_color(0, 255, 0)
                self.viewer.add_geom(left_lane)
                x_left_prev, y_left_prev = x_left, y_left
                
                # Right lane
                x_right, y_right = np.matmul(R,[x_right_rel, y_right_rel])
                x_right = x*m2pixel + x_right + x_track_ini
                y_right = y*m2pixel + y_right + y_track_ini
                right_lane = rendering.Line((x_right_prev, y_right_prev), (x_right, y_right))
                right_lane.set_color(255, 0, 0)
                self.viewer.add_geom(right_lane)
                x_right_prev, y_right_prev = x_right, y_right
    
        # Render current robot position
        x, y, theta = self.sensors
        tras_x = x_track_ini + x*m2pixel
        tras_y = y_track_ini + y*m2pixel
        self.robot_transform.set_translation(tras_x, tras_y)
        self.robot_transform.set_rotation(theta*np.pi/180)
        
        # time.sleep(0.1)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _is_done(self):
        # Check if environment is terminated
        
        # State = (lateral_error, longitudinal_distance)
        # For the straigth lane use case, the lateral error matches the y coordinate
        # and the longitudinal distance matches the x coordinate
        e_lat, _, _, _ = self.state
        long_dist = self.sensors[0]
               
        # Check if environment is terminated
        if e_lat == -2 or e_lat == 2:
            # Robot exceeded circuit boundaries
            done = True
        elif long_dist >= self.x_goal:
            # Robot reached final destination
            done = True
            self.destination_reached = True
        elif self.circuit_number == 2 and long_dist < 0:
            # Robot exceeded circuit boundaries
            done = True
        elif ((e_lat, long_dist) in self.obstacles):
            done = True
        else:
            done = False
        
        return done
    
    def close(self):
        return self.render(close=True)
    
    def _reward(self, action):
        # Calculate reward value
        
        # State = (lateral_error, longitudinal_distance)
        # For the straigth lane use case, the lateral error matches the y coordinate
        # and the longitudinal distance matches the x coordinate
        e_lat, _, _, _ = self.state
        long_dist = self.sensors[0]
        
        # Reward due to robot out of circuit boundaries
        if e_lat == -2 or e_lat == 2 or (self.circuit_number == 2 and long_dist < 0):
            reward_out_boundaries = -1
        else:
            reward_out_boundaries = 0
            
        # Reward due to collision with obstacles
        if ((e_lat, long_dist) in self.obstacles):
            reward_obs = -1
        else:
            reward_obs = 0
        
        # Reward if robot follows the center lane of the circuit
        if e_lat == 0:
            reward_lane = 0
        elif e_lat == -1 or e_lat == 1:
            reward_lane = 0
        else:
            reward_lane = 0
            
        # Reward if robot rotates and does not move forward
        if action[0] == 1:
            reward_action = 0
        else:
            reward_action = 0
            
        # Reward if robot moves forward along the road
        if long_dist > self.long_dist_old:
            reward_move_forward = 0.1
        elif long_dist < self.long_dist_old:
            reward_move_forward = -0.1
        else:
            reward_move_forward = -0.1
        self.long_dist_old = long_dist
            
        # Reward if destination reached
        if self.destination_reached == True:
            reward_destination = 1
        else:
            reward_destination = 0
        
        return reward_lane + reward_out_boundaries + reward_obs + reward_destination + reward_move_forward + reward_action