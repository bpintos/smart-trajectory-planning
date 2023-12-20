import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
# import pandas as pd

class WhiteActionNoise:
    def __init__(self, mean, std_deviation):
        self.mean = mean
        self.std_dev = std_deviation

    def __call__(self, sampled_actions):
        x = np.random.normal(self.mean, self.std_dev)
        return x

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self, sampled_actions):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, target_actor,\
               target_critic, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            next_state_action = tf.keras.layers.concatenate([next_state_batch, target_actions], axis=1)
            y = reward_batch + gamma * target_critic(next_state_action, training=True)
            state_action = tf.keras.layers.concatenate([state_batch, action_batch], axis=1)
            critic_value = critic_model(state_action, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
        print("critic_loss ->  {}".format(critic_loss))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            state_action = tf.keras.layers.concatenate([state_batch, actions], axis=1)
            critic_value = critic_model(state_action, training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))
        print("actor_loss ->  {}".format(actor_loss))

    # We compute the loss and update parameters
    def learn(self, target_actor, target_critic, actor_model, critic_model,\
              actor_optimizer, critic_optimizer, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        if record_range != 0:
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)
    
            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
            
            # Update neural network weights
            self.update(state_batch, action_batch, reward_batch, next_state_batch, target_actor,\
                        target_critic, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_NN(input_layer, hidden_layer, output_layer, hidden_activation = 'relu', output_activation = None):
    initializer = tf.keras.initializers.GlorotNormal()
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_layer))
    for h in hidden_layer:
        model.add(layers.Dense(units=h, activation=hidden_activation, kernel_initializer=initializer))
    model.add(layers.Dense(units=output_layer, activation=output_activation, kernel_initializer=initializer))
    return model

def policy(state, std_dev, actor_model, lower_bound, upper_bound, num_iter):
    # Convert state to tensor
    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    # Evaluate policy
    sampled_actions = tf.squeeze(actor_model(state))
    # Compute exploration noise (Gaussian noise)
    noise = np.random.normal(0, std_dev)
    # frequency = 1
    # noise = std_dev*np.sin(2*np.pi*frequency*num_iter*0.2)
    if not(isinstance(noise, float)):
        if num_iter%2 == 0:
            noise = np.array([0, noise[1]])
        else:
            noise = np.array([noise[0], 0])
    # Action + exploration
    sampled_actions_noise = sampled_actions.numpy() + noise
    # Upper and lower limitation of the action
    action_lim = np.clip(sampled_actions_noise, lower_bound, upper_bound)

    return np.squeeze(action_lim), np.squeeze(sampled_actions), np.squeeze(noise)

def getQvalue(q_table, state, action):
    state_axis = q_table[0]
    index_state = state_axis.index(state)
    action_axis = q_table[1]
    index_action = action_axis.index(action)
    q_values = q_table[2]
    q_value = q_values[index_state, index_action]
    return q_value

def initQtable(env_name, state_variables, actions):
    states = state_variables[0]
    number_variables = len(state_variables)
    for i in range(1,number_variables):
        variable = state_variables[i]
        states_len = len(states)
        states = states*len(variable)
        variable = tuple(map(lambda x: (x,)*states_len, variable))
        variable = tuple(np.hstack(variable))
        states = tuple(map(lambda x, y: tuple(np.hstack((x,y))), states, variable))
    table = np.zeros((len(states),len(actions)))
    return states, actions, table

def getAction(q_table, state):
    state_axis = q_table[0]
    action_axis = q_table[1]
    q_values = q_table[2]
    # Find index of state axis
    index_state = state_axis.index(state)
    # Select q values of actions belonging to that state
    q_values_state = q_values[index_state,:]
    # Find action which maximizes q value
    q_max = np.amax(q_values_state)
    action_index = np.where(q_values_state == q_max)
    action_index = random.choice(action_index[0])
    action = action_axis[action_index]
    return action

def updateQtable(q_table, state, action, q_value_new):
    state_axis = q_table[0]
    index_state = state_axis.index(state)
    action_axis = q_table[1]
    index_action = action_axis.index(action)
    q_values = q_table[2]
    q_values[index_state, index_action] = q_value_new
    return state_axis, action_axis, q_values