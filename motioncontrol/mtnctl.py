# -*- coding: utf-8 -*-

class diffRobotControl:
    def __init__(self, wheelBase, wheelRadius):
        self.wheelBase = wheelBase
        self.wheelRadius = wheelRadius
        
    def long_lat_control(self, linear_vel, angular_vel):
        v_left = linear_vel + angular_vel*self.wheelBase/2
        v_left = v_left/self.wheelRadius
        v_right = linear_vel - angular_vel*self.wheelBase/2
        v_right = v_right/self.wheelRadius
        
        return (v_left, v_right)