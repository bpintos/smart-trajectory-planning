# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:36:31 2020

@author: bpintos
"""

import gym
from gym import spaces
import numpy as np
import time
import sys
from coppelia import sim
from motioncontrol.mtnctl import diffRobotControl

class VehicleTfmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, number_laser, number_IMU, number_GPS, min_distance2robot,\
                 laser_range, wheelBase, wheelRadius, robotLinearVelocity,\
                     robotMaxLinearVelocity, robotMaxAngularVelocity, goal):
        super(VehicleTfmEnv, self).__init__()
        
        # Open communication with Coppelia Sim
        sim.simxFinish(-1) # just in case, close all opened connections
        self.id = sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
        
        # Variable initialization
        self.getLaserSensorFirstTime = [True]*number_laser
        self.getIMUSensorFirstTime = True
        self.getGPSSensorFirstTime = True
        self.getVisionSensorFirstTime = True
        self.sensors = [0]*(number_laser+number_IMU+number_GPS)
        self.sensors_error = [False]*(number_laser+number_IMU+number_GPS)
        self.actuators_error = [False, False]
        self.state = [0]*2#(number_laser+number_IMU)
        self.min_distance2robot = min_distance2robot
        self.laser_range = laser_range
        self.wheelBase = wheelBase
        self.wheelRadius = wheelRadius
        self.robotLinearVelocity = robotLinearVelocity
        self.robotMaxLinearVelocity = robotMaxLinearVelocity
        self.robotMaxAngularVelocity = robotMaxAngularVelocity
        self.goal = goal
        self.robot_control = diffRobotControl(self.wheelBase, self.wheelRadius)
        
        # Action space (normalized between -1 and 1)
        high = np.array([1], dtype = np.float32)
        self.action_space = spaces.Box(-high, high)
        
        # Observation space (normalized between 1 and 0)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype = np.float32)
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1], dtype = np.float32)
        # high = np.array([1, 1], dtype = np.float32)
        # low = np.array([-1, -1], dtype = np.float32)
        self.observation_space = spaces.Box(low, high)
        
        # Scene handles
        resRobot, self.robotHandle = sim.simxGetObjectHandle(self.id,'Pioneer_p3dx',sim.simx_opmode_blocking) # Robot's handle
        if resRobot != sim.simx_return_ok:
            sys.exit('Object Pioneer_p3dx not found in scene')
        resLeftMotor, self.LeftMotorHandle = sim.simxGetObjectHandle(self.id,'Pioneer_p3dx_leftMotor',sim.simx_opmode_blocking) # Left motor's handle
        if resLeftMotor != sim.simx_return_ok:
            sys.exit('Object Pioneer_p3dx_leftMotor not found in scene')
        resRightMotor, self.RightMotorHandle = sim.simxGetObjectHandle(self.id,'Pioneer_p3dx_rightMotor',sim.simx_opmode_blocking) # Right motor's handle
        if resRightMotor != sim.simx_return_ok:
            sys.exit('Object Pioneer_p3dx_rightMotor not found in scene')
        resLaser = []
        self.laserHandle = []
        for i in range(1,number_laser+1):
            res, laser = self.laser1Handle = sim.simxGetObjectHandle(self.id,'Pioneer_p3dx_ultrasonicSensor' + str(i),sim.simx_opmode_blocking) # Laser's handle
            if res != sim.simx_return_ok:
                message = 'Object ' + 'Pioneer_p3dx_ultrasonicSensor' + str(i) + ' not found in scene'
                sys.exit(message)
            resLaser.append(res)
            self.laserHandle.append(laser)
        
    def reset(self):
        # Reset the state of the environment to an initial state
        
        # Stop the simulation
        sim.simxStopSimulation(self.id,sim.simx_opmode_blocking)
        time.sleep(.5)
        
        # Enable the synchronous mode on the client
        sim.simxSynchronous(self.id,True)
        
        # Start the simulation
        sim.simxStartSimulation(self.id,sim.simx_opmode_blocking)
        
        # Reset state
        self.state = self._getState()
        
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        
        # Unnormalize action
        action = action*self.robotMaxAngularVelocity
        robotLinearVelocity = self.robotLinearVelocity*self.robotMaxLinearVelocity
        
        # Calculate actuator signals (motion control)
        v_left, v_right = self.robot_control.long_lat_control(robotLinearVelocity, action)
        
        # Write actuator signals in coppelia sim
        actuators_error = []
        resLeftMotor = sim.simxSetJointTargetVelocity(self.id, self.LeftMotorHandle, v_left, sim.simx_opmode_oneshot)
        if resLeftMotor == sim.simx_return_ok:
            actuators_error.append(False)
        else:
            actuators_error.append(True)
            print('Error left motor response code:', resLeftMotor)
        resRightMotor = sim.simxSetJointTargetVelocity(self.id, self.RightMotorHandle, v_right, sim.simx_opmode_oneshot)
        if resRightMotor == sim.simx_return_ok:
            actuators_error.append(False)
        else:
            actuators_error.append(True)
            print('Error right motor response code:', resRightMotor)
        self.actuators_error = actuators_error
        
        # Execute one time step (one time step is 4*50 = 200 ms)
        steps = 4
        for i in range(steps):
            sim.simxSynchronousTrigger(self.id);
        
        # Get state
        self.state = self._getState()

        return self.state, self._reward(), self._isDone(), (self.sensors_error, self.actuators_error)
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        
        return 0

    def _isDone(self):
        # Check if environment is terminated
        
        lasers = self.sensors[0:16]
        lasers_error = self.sensors_error[0:16]
        positions = self.sensors[19:22]
        positions_error = self.sensors_error[19:22]
        laser_sick_error = self.sensors_error[-1]
        
        # Get state
        state_lasers = self.state[:-1]
        
        # if all(laser_error == False for laser_error in lasers_error) and any(laser <= self.min_distance2robot for laser in lasers):
        #     done = True
        # elif all(position_error == False for position_error in positions_error) and positions[1] >= self.goal:
        #     done = True
        # else:
        #     done = False
        
        if not(laser_sick_error) and any(state_laser*5 <= self.min_distance2robot for state_laser in state_lasers):
            done = True
        elif all(position_error == False for position_error in positions_error) and positions[2] >= self.goal:
            done = True
        else:
            done = False
        
        return done
    
    def _reward(self):
        # Calculate reward value
        
        lasers = self.sensors[0:16]
        lasers_error = self.sensors_error[0:16]
        # velocities = self.sensors[16:19]
        # velocities_error = self.sensors_error[16:19]
        positions = self.sensors[19:22]
        positions_error = self.sensors_error[19:22]
        laser_sick_error = self.sensors_error[-1]
        
        # Get state
        state_lasers = self.state[:-1]
        
        # Penalization if robot crashes into obstacle
        # if all(laser_error == False for laser_error in lasers_error):
        #     for laser in lasers:
        #         reward_obs = reward_obs + np.clip((laser-1)/self.min_distance2robot, -1, 0)
        # else:
        #     reward_obs = 0
        reward_obs = 0
        if not(laser_sick_error):
            for state_laser in state_lasers:
                reward_obs = reward_obs + np.clip((state_laser-0.2)/0.2, -1, 0)
        else:
            reward_obs = 0
        
        # Reward if robot moves forward to destination
        if all(position_error == False for position_error in positions_error):
            reward_dest = 1-abs(positions[2])
        else:
            reward_dest = 0
        
        reward = reward_obs
        
        return reward
    
    def _getSensors(self):
        # Get sensor values from the robot
        
        sensors = []
        sensors_error = []
        # Read laser sensors
        for i in range(0,len(self.laserHandle)):
            # Get raw data from sensor
            if self.getLaserSensorFirstTime[i]:
                raw_data = sim.simxReadProximitySensor(self.id, self.laserHandle[i], sim.simx_opmode_streaming) # Try to retrieve the streamed data
                self.getLaserSensorFirstTime[i] = False
            else:
                raw_data = sim.simxReadProximitySensor(self.id, self.laserHandle[i], sim.simx_opmode_buffer) # Try to retrieve the streamed data
            # Check data feasibility
            if raw_data[0] == sim.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
                if raw_data[1] == True:
                    sensors.append(raw_data[2][2])
                else:
                    sensors.append(self.laser_range)
                sensors_error.append(False)
            else:
                sensors.append(self.laser_range)
                sensors_error.append(True)
        
        # Read IMU sensors
        # Get raw data from sensor
        if self.getIMUSensorFirstTime:
            raw_data = sim.simxGetObjectVelocity(self.id, self.robotHandle, sim.simx_opmode_streaming)
            self.getIMUSensorFirstTime = False
        else:
            raw_data = sim.simxGetObjectVelocity(self.id, self.robotHandle, sim.simx_opmode_buffer)
        # Check data feasibility
        if raw_data[0] == sim.simx_return_ok:
            #linear_vel = np.sqrt(raw_data[1][0]**2 + raw_data[1][1]**2)
            vel_x = raw_data[1][0]
            vel_y = raw_data[1][1]
            vel_angular = -raw_data[2][2]
            sensors.append(vel_x)
            sensors_error.append(False)
            sensors.append(vel_y)
            sensors_error.append(False)
            sensors.append(vel_angular)
            sensors_error.append(False)
        else:
            sensors.append(1)
            sensors_error.append(True)
            sensors.append(1)
            sensors_error.append(True)
            sensors.append(1)
            sensors_error.append(True)
        
        # Read GPS sensor (for robot's positioning, Kalman filter with IMU sensors)
        # Get raw data from sensor
        if self.getGPSSensorFirstTime:
            raw_data_1 = sim.simxGetObjectPosition(self.id, self.robotHandle, -1, sim.simx_opmode_streaming)
            raw_data_2 = sim.simxGetObjectOrientation(self.id, self.robotHandle, -1, sim.simx_opmode_streaming)
            self.getGPSSensorFirstTime = False
        else:
            raw_data_1 = sim.simxGetObjectPosition(self.id, self.robotHandle, -1, sim.simx_opmode_buffer)
            raw_data_2 = sim.simxGetObjectOrientation(self.id, self.robotHandle, -1, sim.simx_opmode_buffer)
        # Check data feasibility
        if raw_data_1[0] == sim.simx_return_ok and raw_data_2[0] == sim.simx_return_ok:
            heading = raw_data_2[1][2]
            x_pos = raw_data_1[1][0]
            y_pos = raw_data_1[1][1]
            sensors.append(heading)
            sensors_error.append(False)
            sensors.append(x_pos)
            sensors_error.append(False)
            sensors.append(y_pos)
            sensors_error.append(False)
        else:
            sensors.append(0)
            sensors_error.append(True)
            sensors.append(0)
            sensors_error.append(True)
            sensors.append(0)
            sensors_error.append(True)
        
        # Read vision sensor
        if self.getVisionSensorFirstTime:
            raw_data = sim.simxGetStringSignal(self.id,'linesDataAtThisTime', sim.simx_opmode_streaming)
            data = sim.simxUnpackFloats(raw_data[1])
            self.getVisionSensorFirstTime = False
        else:
            raw_data = sim.simxGetStringSignal(self.id,'linesDataAtThisTime', sim.simx_opmode_buffer)
            data = sim.simxUnpackFloats(raw_data[1])
        if raw_data[0] == sim.simx_return_ok:
            sensors.append(data)
            sensors_error.append(False)
        else:
            sensors.append(np.zeros(684*2))
            sensors_error.append(True)
        
        # print(sensors)
        # print('vel_x:', sensors[16], ', vel_y:', sensors[17], ', vel_ang', sensors[18], ', heading', sensors[19]*180/np.pi)
        # print('pos_x:', sensors[20])
        return sensors, sensors_error
    
    def _getState(self):
        # Get robot's state for the RL agent
        
        # Get sensor values
        self.sensors, self.sensors_error = self._getSensors()
        
        front_lasers = self.sensors[0:8]
        front_lasers = [np.clip(i/self.laser_range, 0, 1) for i in front_lasers]
        vel_x = np.clip(self.sensors[16]/self.robotMaxLinearVelocity, -1, 1)
        vel_y = np.clip(self.sensors[17]/self.robotMaxLinearVelocity, -1, 1)
        vel_ang = np.clip(self.sensors[18]/self.robotMaxAngularVelocity, -1, 1)
        heading = np.clip(self.sensors[19]/np.pi, -1, 1)
        pos_x = self.sensors[20]
        pos_y = self.sensors[21]
        laser_sick = self.sensors[22]
        
        laser_distances=[]
        for i in range(0,len(laser_sick)-1,2):
           laser_distances.append(np.sqrt((laser_sick[i]-pos_x)**2 + (laser_sick[i+1]-pos_y)**2))
        laser_distances = [laser_distance/5 for laser_distance in laser_distances]
        laser_distances = np.clip(laser_distances[::90],0,1)
        
        distance2target = np.clip((self.goal - self.sensors[20])/8, 0, 1)
        
        state = front_lasers + [vel_x] + [vel_y] + [vel_ang] + [distance2target]
        state = [pos_y] + [heading]
        state = front_lasers + [vel_ang]
        state = list(laser_distances) + [vel_ang]
        
        # print(state)
        return state
    
    def closeEnvironment(self):
        
        # Stop the simulation
        sim.simxStopSimulation(self.id,sim.simx_opmode_blocking)
        
        # Close the communication to CoppeliaSim
        sim.simxFinish(self.id)
        
        return 0