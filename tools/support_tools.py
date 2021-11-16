import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_actor(title, actor_model, state_upper_limit, state_lower_limit):
    ax = plt.axes(projection='3d')
    table_size = (20,20)
    x_axis = np.linspace(state_lower_limit[0], state_upper_limit[0], table_size[0])
    y_axis = np.linspace(state_lower_limit[-1], state_upper_limit[-1], table_size[1])
    z_table = np.zeros(table_size)
    for i_x, x in enumerate(x_axis):
        for i_y, y in enumerate(y_axis):
            state = [x,y]
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            z_table[i_x,i_y] = tf.get_static_value(actor_model(state))
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    ax.plot_surface(x_axis, y_axis, z_table, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('elat')
    ax.set_ylabel('heading')
    ax.set_title(title)
    plt.show()
    return 0

def plot_critic(title, critic_model, state_upper_limit, state_lower_limit, action_upper_limit, action_lower_limit):
    table_size = (10,10,10)
    action_axis = np.linspace(action_lower_limit[0], action_upper_limit[0], table_size[0])
    x_axis = np.linspace(state_lower_limit[0], state_upper_limit[0], table_size[1])
    y_axis = np.linspace(state_lower_limit[-1], state_upper_limit[-1], table_size[2])
    z_table = np.zeros(table_size)
    for i_x, x in enumerate(x_axis):
        for i_y, y in enumerate(y_axis):
            for i_a, a in enumerate(action_axis):
                state_action = [x,y,a]
                state_action = tf.expand_dims(tf.convert_to_tensor(state_action), 0)
                z_table[i_x,i_y,i_a] = tf.get_static_value(critic_model(state_action))
    
    x_axis, action_axis = np.meshgrid(x_axis, action_axis)
    
    for i_y in range(0,len(y_axis),2):
        ax = plt.axes(projection='3d')
        y_value = y_axis[i_y]
        ax.plot_surface(x_axis, action_axis, z_table[:,i_y,:], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('elat')
        ax.set_ylabel('action ang. velocity')
        ax.set_title(title + ' heading = ' + str(y_value*180))
        plt.show()
    return 0