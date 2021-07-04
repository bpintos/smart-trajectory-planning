import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, target_actor, target_critic,\
            actor_model, critic_model, actor_optimizer, critic_optimizer, gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, target_actor, target_critic, actor_model, critic_model,\
              actor_optimizer, critic_optimizer, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, target_actor,\
                    target_critic, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor(num_states, upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    initializer = tf.keras.initializers.GlorotUniform()

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(500, activation="relu", kernel_initializer=initializer)(inputs)
    out = layers.Dense(500, activation="relu", kernel_initializer=initializer)(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=initializer)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states, num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(500, activation="relu")(state_input)
    state_out = layers.Dense(500, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(500, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(500, activation="relu")(concat)
    out = layers.Dense(500, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object, actor_model, lower_bound, upper_bound, ep):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object(sampled_actions)
    # Adding noise to action
    sampled_actions_noise = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions_noise, lower_bound, upper_bound)
    
    # if ep <= 10:
    #     circuit = pd.read_csv('vehiclegym/envs/circuit1_delta_higher_resolution.csv')
    #     delta = circuit['chasis.delta']
    #     action = np.array([delta[index]/20])
    #     action = [np.squeeze(action)]
    # else:
    #     action = [np.squeeze(legal_action)]

    return [np.squeeze(legal_action)], [np.squeeze(sampled_actions)], [np.squeeze(noise)]

def DiscretizeState(state):
    e_lat = state[0]
    e_theta = state[1]
    # Discretize lateral distance
    # if e_lat <= -1:
    #     e_lat_discrete = -1
    # elif e_lat > -1 and e_lat <= -0.6:
    #     e_lat_discrete = -0.8
    # elif e_lat > -0.6 and e_lat <= -0.2:
    #     e_lat_discrete = -0.4
    # elif e_lat > -0.2 and e_lat < 0.2:
    #     e_lat_discrete = 0
    # elif e_lat >= 0.2 and e_lat < 0.6:
    #     e_lat_discrete = 0.4
    # elif e_lat >= 0.6 and e_lat < 1:
    #     e_lat_discrete = 0.8
    # elif e_lat >= 1:
    #     e_lat_discrete = 1
    if e_lat > 0:
        e_lat_discrete = 0.5
    else:
        e_lat_discrete = -0.5
    
    # Discretize vehicle heading
    # if e_theta >= -0.55 and e_theta <= -0.45:
    #     e_theta_discrete = -0.5
    # elif e_theta > -0.45 and e_theta <= -0.35:
    #     e_theta_discrete = -0.4
    # elif e_theta > -0.35 and e_theta <= -0.25:
    #     e_theta_discrete = -0.3
    # elif e_theta > -0.25 and e_theta <= -0.15:
    #     e_theta_discrete = -0.2
    # elif e_theta > -0.15 and e_theta <= -0.05:
    #     e_theta_discrete = -0.1
    # elif e_theta > -0.05 and e_theta < 0.05:
    #     e_theta_discrete = 0
    # elif e_theta >= 0.05 and e_theta < 0.15:
    #     e_theta_discrete = 0.1
    # elif e_theta >= 0.15 and e_theta < 0.25:
    #     e_theta_discrete = 0.2
    # elif e_theta >= 0.25 and e_theta < 0.35:
    #     e_theta_discrete = 0.3
    # elif e_theta >= 0.35 and e_theta < 0.45:
    #     e_theta_discrete = 0.4
    # elif e_theta >= 0.45 and e_theta <= 0.55:
    #     e_theta_discrete = 0.5
    
    e_theta_discrete = 0
        
    return e_lat_discrete, e_theta_discrete

def getQvalue(q_table, state, action):
    state_axis = q_table[0]
    index_state = state_axis.index(state)
    action_axis = q_table[1]
    index_action = action_axis.index(action)
    q_values = q_table[2]
    q_value = q_values[index_state, index_action]
    return q_value

def initQtable():
    actions = (0, 1, 2)
    e_lat_values = (-2, -1, 0, 1, 2)
    long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    states = []
    for e_lat_value in e_lat_values:
        for long_dist_value in long_dist_values:
            states.append((e_lat_value,long_dist_value))
    states = tuple(states)
    # states = tuple(e_lat_values)
    table = np.zeros((len(states),len(actions)))
    #table = np.random.rand(len(states),len(actions))
    return states, actions, table

def initQtable_diff():
    actions = (0, 1, 2)
    e_lat_values = (-2, -1, 0, 1, 2)
    long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    theta_values = (0, 90, 180, -90)
    states = []
    for e_lat_value in e_lat_values:
        for long_dist_value in long_dist_values:
            for theta_value in theta_values:
                states.append((e_lat_value,long_dist_value,theta_value))
    states = tuple(states)
    table = np.zeros((len(states),len(actions)))
    return states, actions, table

def initQtable_2(env_name):
    if env_name == 'vehicle_env_discrete_v1':
        actions = (0, 1, 2)
        e_lat_values = (-2, -1, 0, 1, 2)
        long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        states = []
        for e_lat_value in e_lat_values:
            for long_dist_value in long_dist_values:
                states.append((e_lat_value,long_dist_value))
        states = tuple(states)
        # states = tuple(e_lat_values)
        table = np.zeros((len(states),len(actions)))
        #table = np.random.rand(len(states),len(actions))
    elif env_name == 'diff_env_discrete_v1':
        actions = (0, 1, 2)
        e_lat_values = (-2, -1, 0, 1, 2)
        long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        theta_values = (0, 90, 180, -90)
        states = []
        for e_lat_value in e_lat_values:
            for long_dist_value in long_dist_values:
                for theta_value in theta_values:
                    states.append((e_lat_value,long_dist_value,theta_value))
        states = tuple(states)
        table = np.zeros((len(states),len(actions)))
    elif env_name == 'diff_env_discrete_v2':
        actions = (0, 1, 2)
        e_lat_values = (-2, -1, 0, 1, 2)
        long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        theta_values = (0, 90, 180, -90)
        obs_long_values = (0, 1, 2, 3, 4)
        obs_lat_values = (-1, 0, 1)
        states = []
        for e_lat_value in e_lat_values:
            for long_dist_value in long_dist_values:
                for theta_value in theta_values:
                    for obs_long_value in obs_long_values:
                        for obs_lat_value in obs_lat_values:
                            states.append((e_lat_value,long_dist_value,theta_value,obs_long_value,obs_lat_value))
        states = tuple(states)
        table = np.zeros((len(states),len(actions)))
    else:
        actions = (0, 1, 2)
        e_lat_values = (-2, -1, 0, 1, 2)
        long_dist_values = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        states = []
        for e_lat_value in e_lat_values:
            for long_dist_value in long_dist_values:
                states.append((e_lat_value,long_dist_value))
        states = tuple(states)
        # states = tuple(e_lat_values)
        table = np.zeros((len(states),len(actions)))
        #table = np.random.rand(len(states),len(actions))
    return states, actions, table

def getAction(q_table, state):
    state_axis = q_table[0]
    action_axis = q_table[1]
    q_values = q_table[2]
    index_state = state_axis.index(state)
    q_values_state = q_values[index_state,:]
    action_index = np.argmax(q_values_state)
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

# np.random.seed(7)
# tf.random.set_seed(7)


# table = initQtable()

# state = (-1, -0.4)
# action = -5
# value = getQvalue(table, state, action)
# best_action = getAction(table, state)
# value_best = getQvalue(table, state, best_action)