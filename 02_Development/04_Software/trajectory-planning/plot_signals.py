# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:30:52 2021

@author: bpint
"""
import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()
ax.plot(avg_reward_list_05, color='b', label="std_dev = 0.05")
ax.plot(avg_reward_list_10, color='r', label="std_dev = 0.10")
ax.plot(avg_reward_list_15, color='m', label="std_dev = 0.15")
ax.plot(avg_reward_list_20, color='g', label="std_dev = 0.20")
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Episodic Reward")
ax.grid(color='gray', linestyle='-', linewidth=1)
plt.legend(loc='lower right')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(18, 14))
ax1 = fig.add_subplot(511)
ax2 = fig.add_subplot(512)
ax3 = fig.add_subplot(513)
ax4 = fig.add_subplot(514)
ax5 = fig.add_subplot(515)
plt.rcParams.update({'font.size': 22})
time = np.linspace(0, len(e_lat_record)*0.01, len(e_lat_record))
time_lim = np.linspace(0, len(e_lat_record_lim)*0.01, len(e_lat_record_lim))
ax1.plot(time, e_lat_record, color='b')
ax1.plot(time_lim, e_lat_record_lim, color='r')
ax1.set_title("elat [-]")
ax1.set_yticks(np.arange(-1, 1.1, step=0.5))
ax1.grid(color='gray', linestyle='-', linewidth=1)
ax2.plot(time, e_theta_record, color='b')
ax2.plot(time_lim, e_theta_record_lim, color='r')
ax2.set_title("etheta [-]")
ax2.set_yticks(np.arange(-0.2, 0.3, step=0.1))
ax2.grid(color='gray', linestyle='-', linewidth=1)
ax3.plot(time, steer_record, color='b')
ax3.plot(time_lim, steer_record_lim, color='r')
ax3.set_title("delta [-]")
ax3.set_yticks(np.arange(-1, 1.2, step=0.5))
ax3.grid(color='gray', linestyle='-', linewidth=1)
ax4.plot(time, a_lat_record, color='b')
ax4.plot(time_lim, a_lat_record_lim, color='r')
ax4.set_title("Lateral acceleration [m/s2]")
ax4.set_yticks(np.arange(-10, 11, step=5))
ax4.grid(color='gray', linestyle='-', linewidth=1)
ax5.plot(time, reward_record, color='b')
ax5.plot(time_lim, reward_record_lim, color='r')
ax5.set_title("Episodic Cumulative Reward is ==> {}".format(episodic_reward))
ax5.set_xlabel("Time [s]")
ax5.set_yticks(np.arange(0, 1.2, step=0.2))
ax5.grid(color='gray', linestyle='-', linewidth=1)
fig.tight_layout(pad=1.0)
fig.suptitle('Initial conditions: elat_ini = -0.4 m, etheta_ini = -40 deg', fontsize=30, y=1.02)
plt.show()