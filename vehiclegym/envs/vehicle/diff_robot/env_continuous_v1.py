# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:36:31 2020

Reward baseline must be 0
Single action
State only laser measurements and angular velocity
Only safety requirements

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
    def __init__(self, number_proximity, number_laser, number_IMU, number_GPS, number_laser_rays,\
                 min_distance2robot, laser_range, wheelBase, wheelRadius, robotLinearVelocity,\
                     robotMaxLinearVelocity, robotMaxAngularVelocity):
        super(VehicleTfmEnv, self).__init__()
        
        # Open communication with Coppelia Sim
        sim.simxFinish(-1) # just in case, close all opened connections
        self.id = sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
        
        # Variable initialization
        self.getProxSensorFirstTime = [True]*number_proximity
        self.getIMUSensorFirstTime = True
        self.getGPSSensorFirstTime = True
        self.getVisionSensorFirstTime = True
        self.checkCollisionFirstTime = True
        self.sensors = [0]*(number_proximity + number_laser + number_IMU+number_GPS)
        self.sensors_error = [False]*(number_proximity + number_laser + number_IMU + number_GPS)
        self.actuators_error = [False, False]
        self.state = []
        self.total_number_laser_rays = 684
        self.number_step_laser_rays = round(self.total_number_laser_rays/number_laser_rays)
        if self.total_number_laser_rays%self.number_step_laser_rays == 0:
            self.number_laser_rays = self.total_number_laser_rays//self.number_step_laser_rays
        else:
            self.number_laser_rays = 1 + self.total_number_laser_rays//self.number_step_laser_rays
        self.min_distance2robot = min_distance2robot
        self.laser_range = laser_range
        self.wheelBase = wheelBase
        self.wheelRadius = wheelRadius
        self.robotLinearVelocity = robotLinearVelocity
        self.robotMaxLinearVelocity = robotMaxLinearVelocity
        self.robotMaxAngularVelocity = robotMaxAngularVelocity
        self.goal = (0,0)
        self.robot_control = diffRobotControl(self.wheelBase, self.wheelRadius)
        self.counter_done = 0
        self.destination_reached = False
        
        # Action space (normalized)
        high = np.array([1], dtype = np.float32)
        low = np.array([-1], dtype = np.float32)
        self.action_space = spaces.Box(low, high)
        
        # Observation space (normalized)
        high = np.ones(self.number_laser_rays, dtype = np.float32)
        high = np.append(high, [1])
        low = np.zeros(self.number_laser_rays, dtype = np.float32)
        low = np.append(low, [-1])
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
        for i in range(1,number_proximity+1):
            res, laser = self.laser1Handle = sim.simxGetObjectHandle(self.id,'Pioneer_p3dx_ultrasonicSensor' + str(i),sim.simx_opmode_blocking) # Laser's handle
            if res != sim.simx_return_ok:
                message = 'Object ' + 'Pioneer_p3dx_ultrasonicSensor' + str(i) + ' not found in scene'
                sys.exit(message)
            resLaser.append(res)
            self.laserHandle.append(laser)
        resObs, self.ObsHandle = sim.simxGetObjects(self.id,sim.sim_object_shape_type,sim.simx_opmode_blocking)
        if resObs != sim.simx_return_ok:
            sys.exit('Object Obstacles not found in scene')
        
        self.vel_angular_old = 0
        self.lastTime = 0
        
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
        
        # Reset destination flag
        self.destination_reached = False
        
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        
        # Unnormalize action
        actionLinearVelocity = 0.2*self.robotMaxLinearVelocity
        actionAngularVelocity = action*self.robotMaxAngularVelocity
        
        # Calculate actuator signals (motion control)
        v_left, v_right = self.robot_control.long_lat_control(actionLinearVelocity, actionAngularVelocity)
        
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

        return self.state, self._reward(), self._isDone(), (self.sensors_error, self.actuators_error, self.sensors, self.counter_done, self.destination_reached)
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        
        return 0

    def _isDone(self):

        # front_lasers = self.sensors[0:8]
        # rear_lasers = self.sensors[8:16]
        vel_x = self.sensors[16]
        vel_y = self.sensors[17]
        # vel_ang = self.sensors[18]
        # accel_ang = self.sensors[19]
        # heading = self.sensors[20]
        pos_x = self.sensors[21]
        pos_y = self.sensors[22]
        # laser_sick = self.sensors[23]

        vel = np.sqrt(vel_x**2 + vel_y**2)

        IMUs_error = self.sensors_error[16:20]
        positions_error = self.sensors_error[20:23]
        laser_sick_error = self.sensors_error[-1]
        
        # Get state
        state_lasers = self.state[:self.number_laser_rays]
        
        # Check if episode is terminated
        if not(laser_sick_error) and any(laser*self.laser_range <= self.min_distance2robot for laser in state_lasers):
            done = True
            self.counter_done = 0
        # elif self._checkCollision():
        #     done = True
        #     self.counter_done = 0
        # elif all(position_error == False for position_error in positions_error)\
        #     and pos_x >= self.goal[0]-0.5 and pos_y >= self.goal[1]-0.5\
        #         and pos_x <= self.goal[0]+0.5 and pos_y <= self.goal[1]+0.5:
        #     done = True
        #     self.counter_done += 1
        #     self.destination_reached = True
        elif all(position_error == False for position_error in positions_error)\
            and all(IMU_error == False for IMU_error in IMUs_error)\
                and pos_x >= self.goal[0]-0.5 and pos_y >= self.goal[1]-0.5\
                    and pos_x <= self.goal[0]+0.5 and pos_y <= self.goal[1]+0.5:
                        done = True
                        self.counter_done += 1
                        self.destination_reached = True
        elif all(IMU_error == False for IMU_error in IMUs_error) and vel <= 0:
            done = True
            self.counter_done = 0
        elif all(position_error == False for position_error in positions_error)\
            and pos_y >= 8:
                done = True      
                self.counter_done = 0
        else:
            done = False
        
        return done
    
    def _reward(self):
        # Calculate reward value
        
        # Weights
        k_safety = 1
        k_legal = 0
        k_comfort = 0
        k_task = 0
        
        # Get sensors
        # front_lasers = self.sensors[0:8]
        # rear_lasers = self.sensors[8:16]
        # vel_x = self.sensors[16]
        # vel_y = self.sensors[17]
        # vel_ang = self.sensors[18]
        # accel_ang = self.sensors[19]
        # heading = self.sensors[20]
        # pos_x = self.sensors[21]
        # pos_y = self.sensors[22]
        # laser_sick = self.sensors[23]
        
        # Get errors from sensors
        IMUs_error = self.sensors_error[16:20]
        # positions_error = self.sensors_error[20:23]
        laser_sick_error = self.sensors_error[-1]
        
        # Get state
        state_lasers = self.state[:self.number_laser_rays]
        # vel_norm_lin = self.state[self.number_laser_rays]
        # vel_norm_ang = self.state[self.number_laser_rays + 1]
        # accel_ang_norm = self.state[self.number_laser_rays + 2]
        # distance2target = self.state[self.number_laser_rays + 3]
        # print('distance2target: ', distance2target)
        
        # Local variables
        # vel = np.sqrt(vel_x**2 + vel_y**2)
        # vel = vel_norm_lin*self.robotMaxLinearVelocity
        # accel_ang = accel_ang_norm*0.005
        
        # Penalization if robot crashes into obstacle (Safety requirements)
        min_distance = 1
        if not(laser_sick_error):
            min_state_laser = min(state_lasers)
            reward_safety = np.clip((min_state_laser*self.laser_range - min_distance)/(min_distance - self.min_distance2robot), -1, 0)
        else:
            reward_safety = 0
        
        # # Penalization if angular acceleration is too high (Comfort requirements)
        # accel_max = 0.001
        # if all(IMU_error == False for IMU_error in IMUs_error):
        #     reward_comfort = np.clip((accel_max - abs(accel_ang))/0.0020, -1, 0)
        # else:
        #     reward_comfort = 0
        
        # # Penalization if robot exceeds maximum velocity (Legal requirements)
        # if distance2target >= 0.25:
        #     vel_max = 0.2
        # else:
        #     vel_max = 0
        # if all(IMU_error == False for IMU_error in IMUs_error):
        #     reward_legal = np.clip((vel_max - vel)/(1-vel_max), -1, 0)
        # else:
        #     reward_legal = 0
        
        # # Penalization if the robot does not fulfil the desired task (Task oriented requirements)
        # # Task 1: Robot should reach destination in the minimum time required (Energy requirements)
        # if distance2target >= 0.25:
        #     vel_target = 0.2
        # else:
        #     vel_target = 0
        # if all(IMU_error == False for IMU_error in IMUs_error):
        #     reward_task_1 = np.clip((vel - vel_target)/vel_target, -1, 0)
        # else:
        #     reward_task_1 = 0
        
        # reward_task = reward_task_1
        
        # # Print reward values
        # print ('Linear velocity:', vel)
        # print ('Target linear velocity:', vel_target)
        print ('Reward safety:', reward_safety)
        # # print ('Reward comfort:', reward_comfort)
        # print ('Reward legal:', reward_legal)
        # print ('Reward task:', reward_task)
        
        # Final reward
        reward = k_safety*reward_safety
        
        # print ('Final reward:', reward)
        
        return reward
    
    def _getSensors(self):
        # Get sensor values from the robot
        
        sensors = []
        sensors_error = []
        # Read laser sensors
        for i in range(0,len(self.laserHandle)):
            # Get raw data from sensor
            if self.getProxSensorFirstTime[i]:
                raw_data = sim.simxReadProximitySensor(self.id, self.laserHandle[i], sim.simx_opmode_streaming) # Try to retrieve the streamed data
                self.getProxSensorFirstTime[i] = False
            else:
                raw_data = sim.simxReadProximitySensor(self.id, self.laserHandle[i], sim.simx_opmode_buffer) # Try to retrieve the streamed data
            # Check data feasibility
            if raw_data[0] == sim.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
                if raw_data[1] == True:
                    sensors.append(raw_data[2][2])
                else:
                    sensors.append(2)
                sensors_error.append(False)
            else:
                sensors.append(2)
                sensors_error.append(True)
        
        # Read IMU sensors
        # Get raw data from sensor
        if self.getIMUSensorFirstTime:
            raw_data = sim.simxGetObjectVelocity(self.id, self.robotHandle, sim.simx_opmode_streaming)
            currentTime = sim.simxGetLastCmdTime(self.id)
            self.getIMUSensorFirstTime = False
        else:
            raw_data = sim.simxGetObjectVelocity(self.id, self.robotHandle, sim.simx_opmode_buffer)
            currentTime = sim.simxGetLastCmdTime(self.id)
        # Check data feasibility
        if raw_data[0] == sim.simx_return_ok:
            #linear_vel = np.sqrt(raw_data[1][0]**2 + raw_data[1][1]**2)
            vel_x = raw_data[1][0]
            vel_y = raw_data[1][1]
            vel_angular = -raw_data[2][2]
            dt = currentTime-self.lastTime
            accel_angular = -(vel_angular-self.vel_angular_old)/dt
            sensors.append(vel_x)
            sensors_error.append(False)
            sensors.append(vel_y)
            sensors_error.append(False)
            sensors.append(vel_angular)
            sensors_error.append(False)
            sensors.append(accel_angular)
            sensors_error.append(False)
            self.vel_angular_old = vel_angular
            self.lastTime = currentTime
        else:
            sensors.append(1)
            sensors_error.append(True)
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
            sensors.append(np.zeros(self.total_number_laser_rays*2))
            sensors_error.append(True)
        
        # print(sensors)
        # print('vel_x:', sensors[16], ', vel_y:', sensors[17], ', vel_ang', sensors[18], ', heading', sensors[19]*180/np.pi)
        # print('pos_x:', sensors[20])
        return sensors, sensors_error
    
    def _getState(self):
        # Get robot's state for the RL agent
        
        # Get sensor values
        self.sensors, self.sensors_error = self._getSensors()
        
        # front_lasers = self.sensors[0:8]
        # rear_lasers = self.sensors[8:16]
        vel_x = self.sensors[16]
        vel_y = self.sensors[17]
        vel_ang = self.sensors[18]
        accel_ang = self.sensors[19]
        # heading = self.sensors[20]
        pos_x = self.sensors[21]
        pos_y = self.sensors[22]
        laser_sick = self.sensors[23]

        # front_norm_lasers = [np.clip(i/2, 0, 1) for i in front_lasers]
        # vel_norm_x = np.clip(vel_x/self.robotMaxLinearVelocity, -1, 1)
        # vel_norm_y = np.clip(vel_y/self.robotMaxLinearVelocity, -1, 1)
        vel_norm_ang = np.clip(vel_ang/self.robotMaxAngularVelocity, -1, 1)
        vel = np.sqrt(vel_x**2 + vel_y**2)
        vel_norm_lin = np.clip(vel/self.robotMaxLinearVelocity, 0, 1)
        accel_ang_norm = np.clip(accel_ang/0.005, -1, 1)
        
        laser_distances=[]
        for i in range(0,len(laser_sick)-1,2):
           laser_distances.append(np.sqrt((laser_sick[i]-pos_x)**2 + (laser_sick[i+1]-pos_y)**2))
        laser_distances = [laser_distance/self.laser_range for laser_distance in laser_distances]
        laser_distances = np.clip(laser_distances[::self.number_step_laser_rays],0,1)
        
        distance2target = np.clip(np.sqrt((self.goal[0]-pos_x)**2 + (self.goal[1]-pos_y)**2)/2, 0, 1)
        # print('distance2target: ', distance2target)
        
        # state = front_norm_lasers + [vel_norm_x] + [vel_norm_y] + [vel_norm_ang] + [distance2target]
        state = list(laser_distances) + [vel_norm_ang]
        # state = list(laser_distances) + [vel_norm_lin] + [vel_norm_ang] + [accel_ang_norm] + [distance2target]
        
        return state
    
    def _checkCollision(self):
        CollisionVector = []
        Collision = False
        if self.checkCollisionFirstTime:
            for obs in self.ObsHandle:
                res, collision = sim.simxCheckCollision(self.id,self.robotHandle,obs,sim.simx_opmode_streaming)
                CollisionVector.append(collision)
        else:
            for obs in self.ObsHandle:
                res, collision = sim.simxCheckCollision(self.id,self.robotHandle,obs,sim.simx_opmode_buffer)
                CollisionVector.append(collision)
        Collision = any(x == True for x in CollisionVector)
        
        return Collision
    
    def setTargetDestination(self, firstRun, destinations):
        
        if firstRun:
            self.goal = destinations[0]
        else:
            idx = destinations.index(self.goal)
            idx += 1
            idx = idx%len(destinations)
            self.goal = destinations[idx]
            
        self.destination_reached = False
        
        return 0
        
    def closeEnvironment(self):
        
        # Stop the simulation
        sim.simxStopSimulation(self.id,sim.simx_opmode_blocking)
        
        # Close the communication to CoppeliaSim
        sim.simxFinish(self.id)
        
        return 0
