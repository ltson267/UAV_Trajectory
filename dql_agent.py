import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import numpy as np
from config import *
from replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    TensorFlow 2.x DQN Agent with:
    - Dueling architecture (separate Value and Advantage streams)
    - Double DQN (action selection from main, evaluation from target)
    - Prioritized Experience Replay (PER) with importance sampling
    - Soft target updates (TAU) + periodic hard updates
    """
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        
        # Training parameters
        self.target_update_freq = 200  # Hard update frequency
        self.training_step = 0
        self.clip_norm = 1.0
        self.warmup_samples = self.batch_size * 10
        
        # Build networks
        self._build_model()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clip_norm)

    def _build_dueling_network(self, name):
        """
        Build Dueling DQN architecture:
        Input → FC layers → Split into Value/Advantage → Combine to Q-values
        """
        inputs = Input(shape=(self.state_dim,), name=f'{name}_input')
        
        # Shared layers
        fc1 = layers.Dense(256, activation='relu', 
                          kernel_initializer='he_normal',
                          name=f'{name}_fc1')(inputs)
        fc2 = layers.Dense(128, activation='relu',
                          kernel_initializer='he_normal',
                          name=f'{name}_fc2')(fc1)
        
        # Value stream
        value_fc = layers.Dense(64, activation='relu',
                               kernel_initializer='he_normal',
                               name=f'{name}_value_fc')(fc2)
        value = layers.Dense(1, name=f'{name}_value')(value_fc)
        
        # Advantage stream
        advantage_fc = layers.Dense(64, activation='relu',
                                   kernel_initializer='he_normal',
                                   name=f'{name}_advantage_fc')(fc2)
        advantage = layers.Dense(self.action_dim, 
                                name=f'{name}_advantage')(advantage_fc)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        def dueling_combine(inputs_list):
            val, adv = inputs_list
            mean_adv = tf.reduce_mean(adv, axis=1, keepdims=True)
            return val + (adv - mean_adv)
        
        q_values = layers.Lambda(dueling_combine, 
                                name=f'{name}_q_values')([value, advantage])
        
        return Model(inputs=inputs, outputs=q_values, name=name)

    def _build_model(self):
        """Build main and target networks"""
        self.model = self._build_dueling_network('main_network')
        self.target_model = self._build_dueling_network('target_network')
        
        # Initialize target with main weights
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def _train_step(self, states, targets, is_weights):
        """
        Single training step with gradient computation
        Uses Huber loss weighted by importance sampling
        """
        with tf.GradientTape() as tape:
            # Forward pass
            q_values = self.model(states, training=True)
            
            # Huber loss (delta = 1.0)
            delta = targets - q_values
            abs_delta = tf.abs(delta)
            quadratic_part = tf.minimum(abs_delta, 1.0)
            linear_part = abs_delta - quadratic_part
            per_sample_loss = tf.reduce_sum(
                0.5 * quadratic_part**2 + linear_part, 
                axis=1
            )
            
            # Weight by importance sampling
            weighted_loss = per_sample_loss * is_weights
            loss = tf.reduce_mean(weighted_loss)
        
        # Compute and apply gradients (clipnorm handled by optimizer)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_batch = np.expand_dims(state, axis=0)
        q_vals = self.model.predict(state_batch, verbose=0)[0]
        return np.argmax(q_vals)

    def train(self):
        """
        Train on batch from PER:
        1. Sample with priorities
        2. Compute Double DQN targets
        3. Calculate TD errors BEFORE training
        4. Train network
        5. Update PER priorities
        6. Update target network (soft + periodic hard)
        """
        # Warmup phase
        if len(self.memory) < self.warmup_samples:
            return
        
        # Sample from PER
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.memory.sample(self.batch_size)
        
        # Clip rewards for stability
        norm_rewards = np.clip(rewards, -10.0, 10.0)
        
        # Double DQN: action selection from main, evaluation from target
        next_q_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_main, axis=1)
        
        target_q = self.target_model.predict(next_states, verbose=0)
        current_q = self.model.predict(states, verbose=0)
        
        # Compute targets and TD errors BEFORE training
        targets = current_q.copy()
        td_errors = []
        
        for i in range(self.batch_size):
            if dones[i]:
                target_val = norm_rewards[i]
            else:
                target_val = norm_rewards[i] + self.gamma * target_q[i, best_actions[i]]
            targets[i, actions[i]] = target_val
            td_errors.append(target_val - current_q[i, actions[i]])
        
        td_errors = np.array(td_errors)
        
        # Train the network
        loss = self._train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(targets, dtype=tf.float32),
            tf.convert_to_tensor(is_weights, dtype=tf.float32)
        )
        
        # Update PER priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Update target network
        self.training_step += 1
        
        # Hard update every N steps
        if self.training_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        # Soft update every step
        self._soft_update_target()

    def _soft_update_target(self):
        """Soft update: θ_target = τ*θ_main + (1-τ)*θ_target"""
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        updated_weights = []
        for main_w, target_w in zip(main_weights, target_weights):
            updated_weights.append(TAU * main_w + (1.0 - TAU) * target_w)
        
        self.target_model.set_weights(updated_weights)
