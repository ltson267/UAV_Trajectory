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
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.memory = ReplayBuffer()
        self._build_model()

    def _build_model(self):
        self.states = tf.placeholder(tf.float32, [None, self.state_dim])
        self.targets = tf.placeholder(tf.float32, [None, self.action_dim])  #Truyền vào mạng Q mục tiêu

        fc1 = tf.layers.dense(self.states, 64, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
        self.q_values = tf.layers.dense(fc2, self.action_dim)

        self.loss = tf.reduce_mean(tf.square(self.targets - self.q_values))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_vals = self.sess.run(self.q_values, {self.states: [state]})
        return np.argmax(q_vals[0])

    def train(self):
        if len(self.memory) < self.batch_size:  #Do ban đầu chưa thử sai nên memory batch rỗng nên chưa train()
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_next = self.sess.run(self.q_values, {self.states: next_states})
        q_target = self.sess.run(self.q_values, {self.states: states})
        for i in range(self.batch_size):
            q_target[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * np.max(q_next[i])
        self.sess.run(self.optimizer, {self.states: states, self.targets: q_target})
