import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Prioritized Experience Replay (PER)
class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        size = len(self.buffer)
        prios = self.priorities[:size]
        probs = prios ** self.alpha
        probs /= probs.sum() + 1e-12
        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() + 1e-12
        self.beta = min(1.0, self.beta + self.beta_increment)
        state, action, reward, next_state, done = map(np.array, zip(*samples))
        return state, action, reward, next_state, done, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)
