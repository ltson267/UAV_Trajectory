import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from config import *
from replay_buffer import PrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        # Prioritized Experience Replay for improved sample efficiency
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        # Use soft updates by default; keep periodic hard update as fallback
        self.target_update_freq = 200
        self.training_step = 0
        self.clip_norm = 1.0     # Gradient clipping norm
        self.warmup_samples = self.batch_size * 10  # Warmup before training
        self._build_model()

    def _build_model(self):
        # Main Q-Network (Dueling Architecture)
        self.states = tf.placeholder(tf.float32, [None, self.state_dim])
        self.targets = tf.placeholder(tf.float32, [None, self.action_dim])

        with tf.variable_scope('main_network'):
            fc1 = tf.layers.dense(self.states, 256, activation=tf.nn.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(), name='fc1')
            fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(), name='fc2')

            # Dueling: Split into Value and Advantage streams
            value_fc = tf.layers.dense(fc2, 64, activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.he_normal(), name='value_fc')
            advantage_fc = tf.layers.dense(fc2, 64, activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.he_normal(), name='advantage_fc')

            # Value stream (state value)
            self.value = tf.layers.dense(value_fc, 1, name='value')

            # Advantage stream (action advantages)
            self.advantage = tf.layers.dense(advantage_fc, self.action_dim, name='advantage')

            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            mean_advantage = tf.reduce_mean(self.advantage, axis=1, keepdims=True)
            self.q_values = self.value + (self.advantage - mean_advantage)

        # Target Q-Network (Dueling Architecture)
        with tf.variable_scope('target_network'):
            target_fc1 = tf.layers.dense(self.states, 256, activation=tf.nn.relu,
                                         kernel_initializer=tf.keras.initializers.he_normal(), name='fc1')
            target_fc2 = tf.layers.dense(target_fc1, 128, activation=tf.nn.relu,
                                         kernel_initializer=tf.keras.initializers.he_normal(), name='fc2')

            # Dueling target streams
            target_value_fc = tf.layers.dense(target_fc2, 64, activation=tf.nn.relu,
                                              kernel_initializer=tf.keras.initializers.he_normal(), name='value_fc')
            target_advantage_fc = tf.layers.dense(target_fc2, 64, activation=tf.nn.relu,
                                                  kernel_initializer=tf.keras.initializers.he_normal(), name='advantage_fc')

            target_value = tf.layers.dense(target_value_fc, 1, name='value')
            target_advantage = tf.layers.dense(target_advantage_fc, self.action_dim, name='advantage')

            mean_target_advantage = tf.reduce_mean(target_advantage, axis=1, keepdims=True)
            self.target_q_values = target_value + (target_advantage - mean_target_advantage)

        # Huber Loss (robust), with importance-sampling weights support
        self.is_weights = tf.placeholder(tf.float32, [None])  # PER importance weights
        delta = self.targets - self.q_values
        abs_delta = tf.abs(delta)
        quadratic_part = tf.clip_by_value(abs_delta, 0.0, 1.0)
        linear_part = abs_delta - quadratic_part
        per_sample_loss = tf.reduce_sum(0.5 * quadratic_part**2 + linear_part, axis=1)
        weighted_loss = per_sample_loss * self.is_weights
        self.loss = tf.reduce_mean(weighted_loss)

        # Single optimizer definition with gradient clipping
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer = self.optimizer.apply_gradients(zip(gradients, variables))

        # Target network update operation (hard copy)
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main_network')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
        self.update_target = [target_vars[i].assign(main_vars[i]) for i in range(len(main_vars))]
        # Soft update ops using TAU
        self.soft_update_target = [
            target_vars[i].assign(TAU * main_vars[i] + (1.0 - TAU) * target_vars[i])
            for i in range(len(main_vars))
        ]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Initialize target network with main network weights
        self.sess.run(self.update_target)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_vals = self.sess.run(self.q_values, {self.states: [state]})[0]
        return np.argmax(q_vals)

    def train(self):
        # Warmup phase: collect enough samples before training
        if len(self.memory) < self.warmup_samples:
            return

        states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)

        # Simple fixed clipping (remove running normalization to reduce drift)
        norm_rewards = np.clip(rewards, -5.0, 5.0)

        # Double DQN: Use main network to select actions, target network to evaluate
        # Get best actions from main network
        next_q_values_main = self.sess.run(self.q_values, {self.states: next_states})
        best_actions = np.argmax(next_q_values_main, axis=1)

        # Get Q-values for those actions from target network
        target_q_values = self.sess.run(self.target_q_values, {self.states: next_states})
        current_q_values = self.sess.run(self.q_values, {self.states: states})

        # Compute target Q-values using Double DQN
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = norm_rewards[i]
            else:
                targets[i, actions[i]] = norm_rewards[i] + self.gamma * target_q_values[i, best_actions[i]]

        # Compute TD errors for PER priority updates
        td_errors = []
        for i in range(self.batch_size):
            td_errors.append(targets[i, actions[i]] - self.sess.run(self.q_values, {self.states: states[i:i+1]})[0, actions[i]])
        td_errors = np.array(td_errors)

        # Train the main network with Huber loss and gradient clipping
        self.sess.run(self.optimizer, {self.states: states, self.targets: targets, self.is_weights: is_weights})

        # Update PER priorities
        self.memory.update_priorities(indices, td_errors)

        # Update target network periodically (less frequently to prevent forgetting)
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.sess.run(self.update_target)
        # Soft update every step for smoother tracking
        self.sess.run(self.soft_update_target)

        # Decay epsilon (disabled to let training script handle it)
        pass
