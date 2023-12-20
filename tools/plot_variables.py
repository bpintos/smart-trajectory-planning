#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bpintos
"""

import matplotlib.pyplot as plt
import numpy as np

pos_x = [x[22] for x in sensors_record]
pos_y = [x[23] for x in sensors_record]

fig = plt.figure(figsize=(18, 18))
ax1 = fig.add_subplot(111)
ax1.plot(pos_x, pos_y, color='b', linewidth=10)
ax1.set_xticks(np.arange(-5, 5.1, step=1))
ax1.set_yticks(np.arange(-5, 5.1, step=1))
plt.show()

fig = plt.figure(figsize=(18, 14))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

plt.rcParams.update({'font.size': 22})
time_plot = np.linspace(0, len(action_record)*0.2, len(action_record))
# vel_ang = list(map(lambda x: x[1], action_record))
ax1.plot(time_plot, action_record, 'b')
ax1.set_title('Evolution of normalized angular velocity over time')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Normalized velocity [-]')
ax1.set_yticks(np.arange(-1, 1.1, step=0.5))
ax1.grid(color='gray', linestyle='-', linewidth=1)
accel_ang_record = list(map(lambda x: x[19], sensors_record))
ax2.plot(time_plot, accel_ang_record, 'b')
ax2.set_title('Evolution of angular acceleration over time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Acceleration [rad/s2]')
ax2.set_yticks(np.arange(-0.003, 0.0031, step=0.001))
ax2.grid(color='gray', linestyle='-', linewidth=1)
fig.tight_layout(pad=1.0)
# fig.suptitle('', fontsize=30, y=1.02)
plt.show()

fig = plt.figure(figsize=(18, 14))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

plt.rcParams.update({'font.size': 22})
time_plot = np.linspace(0, len(action_record)*0.2, len(action_record))
vel_ang = list(map(lambda x: x[1], action_record))
ax1.plot(time_plot, vel_ang, 'b')
ax1.set_title('Vel ang [-]')
ax1.set_yticks(np.arange(-1, 1.1, step=0.5))
ax1.grid(color='gray', linestyle='-', linewidth=1)
vel_lin = list(map(lambda x: x[0], action_record))
ax2.plot(time_plot, vel_lin, 'b')
ax2.set_title('Vel lin [-]')
ax2.set_yticks(np.arange(0, 1.1, step=0.2))
ax2.grid(color='gray', linestyle='-', linewidth=1)
fig.tight_layout(pad=1.0)
# fig.suptitle('', fontsize=30, y=1.02)
plt.show()

fig = plt.figure(figsize=(18, 14))
ax1 = fig.add_subplot(111)

plt.rcParams.update({'font.size': 22})
ax1.plot(episodic_reward_record_param4, 'b', label='Multiple actions. Safety req.')
ax1.plot(episodic_reward_record_param5, 'r', label='Multiple actions. Safety+legal req.')
ax1.plot(episodic_reward_record_param6, 'g', label='Multiple actions. Safety+legal+comfort req.')
ax1.plot(episodic_reward_record_param7, 'k', label='Multiple actions. Safety+legal+comfort+task-oriented req.')
ax1.set_title('Episodic reward')
ax1.set_yticks(np.arange(-1.5, 0.1, step=0.5))
ax1.grid(color='gray', linestyle='-', linewidth=1)
ax1.legend(loc='lower right')
plt.show()