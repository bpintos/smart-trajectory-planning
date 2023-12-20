#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:11:36 2021

@author: bpintos
"""

import os
import sys

mode = sys.argv[1]
scene = sys.argv[2]

os.chdir('../../CoppeliaSim')
# os.system('./coppeliaSim.sh /home/bpintos/Projects/smart-trajectory-planning/vehiclegym/envs/scene/PioneerScene.ttt')
# os.system('./coppeliaSim.sh /home/bpintos/Projects/smart-trajectory-planning/vehiclegym/envs/scene/b1_scene.ttt')
# os.system('./coppeliaSim.sh /home/bpintos/Projects/smart-trajectory-planning/vehiclegym/envs/scene/Pionner_room_obstacles.ttt')

scene_command = './coppeliaSim.sh ' + mode + ' /home/bpintos/Projects/smart-trajectory-planning/vehiclegym/envs/scene/'\
    + scene

print(scene_command)
os.system(scene_command)