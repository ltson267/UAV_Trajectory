import numpy as np
import random
from config import *

class UAVEnv:
    def __init__(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.grid_x, self.grid_y = GRID_SIZE
        self.actions = ACTIONS
        self.reset()

    def reset(self):
        self.uav_pos = np.array([0.0, 0.0, HEIGHT_LEVELS[0]])
        self.destination = np.array([self.grid_x - 1, self.grid_y - 1, HEIGHT_LEVELS[0]])

        grid_x, grid_y = self.grid_x, self.grid_y  
        self.users = []
        self.cluster_centers = []
        min_dist = 2 * CLUSTER_RADIUS + 0.5
        attempts = 0

        while len(self.cluster_centers) < NUM_CLUSTERS and attempts < 1000:
            cx = random.uniform(CLUSTER_RADIUS, grid_x - CLUSTER_RADIUS)
            cy = random.uniform(CLUSTER_RADIUS, grid_y - CLUSTER_RADIUS)
            center = np.array([cx, cy])
            if all(np.linalg.norm(center - c) >= min_dist for c in self.cluster_centers):
                self.cluster_centers.append(center)
            attempts += 1
        if len(self.cluster_centers) < NUM_CLUSTERS:
            raise RuntimeError("Error generating cluster centers")
        
        for center in self.cluster_centers:
            cluster = []
            for _ in range(USERS_PER_CLUSTER):
                while True:
                    dx = random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
                    dy = random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
                    if dx**2 + dy**2 <= CLUSTER_RADIUS**2:
                        ux = center[0] + dx
                        uy = center[1] + dy
                        cluster.append(np.array([ux, uy]))
                        break
            self.users.append(cluster)

        self.collected = np.zeros(NUM_CLUSTERS)
        self.hover_steps = np.zeros(NUM_CLUSTERS)
        self.steps = 0
        self.trajectory = [self.uav_pos.copy()]
        return self.get_state()

    def get_state(self):
        return np.array([self.uav_pos[0], self.uav_pos[1]], dtype=np.float32)

    def step(self, action_idx):
        action = self.actions[action_idx]
        if action == "left":
            self.uav_pos[0] = max(0, self.uav_pos[0] - 1)
        elif action == "right":
            self.uav_pos[0] = min(self.grid_x - 1, self.uav_pos[0] + 1)
        elif action == "forward":
            self.uav_pos[1] = min(self.grid_y - 1, self.uav_pos[1] + 1)
        elif action == "backward":
            self.uav_pos[1] = max(0, self.uav_pos[1] - 1)
      
        self.last_action = action
        reward = self.calculate_reward()
        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())
        done = self.is_done()
        return self.get_state(), reward, done

    def calculate_reward(self):
        reward = 0
        hover_bonus = 5
        hover_penalty = 5
        max_hover = 5
        in_any_cluster = False
        for m, center in enumerate(self.cluster_centers):
            dist = np.sqrt((self.uav_pos[0] - center[0])**2 + (self.uav_pos[1] - center[1])**2)
            if dist <= CLUSTER_RADIUS:
                in_any_cluster = True
                if self.collected[m] == 0:
                    reward += 100 
                    self.collected[m] = 1
                
                if hasattr(self, 'last_action') and self.last_action == 'hover':
                    self.hover_steps[m] += 1
                    if self.hover_steps[m] <= max_hover:
                        reward += hover_bonus
                    else:
                        reward -= hover_penalty
                else:
                    self.hover_steps[m] = 0 
            else:
                self.hover_steps[m] = 0

        reward -= 1 #Tiết kiệm pin
        
        if np.allclose(self.uav_pos, self.destination, atol=1e-1):
            reward += 1000
        return reward

    def is_done(self):
        return np.allclose(self.uav_pos, self.destination, atol=1e-1) or self.steps >= MAX_STEPS

    def get_trajectory(self):
        return np.array(self.trajectory)

    def get_users(self):
        return np.array(self.users).reshape(-1, 2)
