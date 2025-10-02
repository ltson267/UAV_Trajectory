import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from config import *
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.memory = ReplayBuffer()
        self.target_update_freq = 50
        self.training_step = 0
        self._build_model()

    def _build_model(self):
        # Main Q-Network
        self.states = tf.placeholder(tf.float32, [None, self.state_dim])
        self.targets = tf.placeholder(tf.float32, [None, self.action_dim])
        
        with tf.variable_scope('main_network'):
            fc1 = tf.layers.dense(self.states, 128, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
            fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu, name='fc3')
            self.q_values = tf.layers.dense(fc3, self.action_dim, name='q_values')

        # Target Q-Network (separate network)
        with tf.variable_scope('target_network'):
            target_fc1 = tf.layers.dense(self.states, 128, activation=tf.nn.relu, name='fc1')
            target_fc2 = tf.layers.dense(target_fc1, 128, activation=tf.nn.relu, name='fc2')
            target_fc3 = tf.layers.dense(target_fc2, 64, activation=tf.nn.relu, name='fc3')
            self.target_q_values = tf.layers.dense(target_fc3, self.action_dim, name='q_values')

        # Loss and optimizer
        self.loss = tf.reduce_mean(tf.square(self.targets - self.q_values))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # Target network update operation
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main_network')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
        self.update_target = [target_vars[i].assign(main_vars[i]) for i in range(len(main_vars))]
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Initialize target network with main network weights
        self.sess.run(self.update_target)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_vals = self.sess.run(self.q_values, {self.states: [state]})
        return np.argmax(q_vals[0])

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Use target network for stable Q-targets
        target_q_values = self.sess.run(self.target_q_values, {self.states: next_states})
        current_q_values = self.sess.run(self.q_values, {self.states: states})
        
        # Compute target Q-values
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(target_q_values[i])
        
        # Train the main network
        self.sess.run(self.optimizer, {self.states: states, self.targets: targets})
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.sess.run(self.update_target)
        
        # Decay epsilon (disabled to let training script handle it)
        pass
