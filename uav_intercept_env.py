import numpy as np
import random
from config import *

class UAVInterceptEnv:
    def __init__(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.grid_x, self.grid_y = GRID_SIZE
        self.actions = ACTIONS
        self.battery_levels = [2, 1, 0]
        self.max_steps = MAX_STEPS
        self.num_su = NUM_SU
        self.num_du = NUM_DU
        self.reset()

    def reset(self):
        self.uav_pos = np.array([0.0, 0.0])
        self.battery_level = 2
        self.steps = 0

        su_x = np.linspace(2, self.grid_x - 2, self.num_su)
        du_x = np.linspace(2, self.grid_x - 2, self.num_du)
        self.su_nodes = [np.array([x, 2]) for x in su_x]
        self.du_nodes = [np.array([x, self.grid_y - 2]) for x in du_x]

        self.connections = []
        for su in self.su_nodes:
            dists = [np.linalg.norm(su - du) for du in self.du_nodes]
            min_idx = np.argmin(dists)
            du = self.du_nodes[min_idx]
            self.connections.append((su, du))
        self.collected = np.zeros(len(self.connections))
        self.trajectory = [self.uav_pos.copy()]
        return self.get_state()

    def get_state(self):
        return np.concatenate([self.uav_pos, [self.battery_level], self.collected], axis=0)

    def step(self, action_idx):
        action = self.actions[action_idx]
        if self.battery_level == 2:
            move_step = 1.0
        elif self.battery_level == 1:
            move_step = 0.5
        else:
            move_step = 0.2

        if action == "left":
            self.uav_pos[0] = max(0, self.uav_pos[0] - move_step)
        elif action == "right":
            self.uav_pos[0] = min(self.grid_x - 1, self.uav_pos[0] + move_step)
        elif action == "forward":
            self.uav_pos[1] = min(self.grid_y - 1, self.uav_pos[1] + move_step)
        elif action == "backward":
            self.uav_pos[1] = max(0, self.uav_pos[1] - move_step)

        reward = 0
        collected_this_step = 0

        if action == "hover":
            for i, (su, du) in enumerate(self.connections):
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                if dist < 0.5 and self.collected[i] == 0:
                    reward += 100
                    self.collected[i] = 1
                    collected_this_step += 1
            if collected_this_step == 0:
                reward -= 2

        reward -= 0.5

        if collected_this_step > 1:
            reward += 30 * (collected_this_step - 1)

        if np.all(self.collected == 1):
            reward += 1000

        if self.steps >= self.max_steps and not np.all(self.collected == 1):
            reward -= 50

        if self.battery_level > 0:
            self.battery_level -= 1 if self.steps % 20 == 0 else 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())
        done = self.is_done()
        return self.get_state(), reward, done

    def _point_to_segment_dist(self, p, a, b):
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def is_done(self):
        return np.all(self.collected == 1) or self.steps >= self.max_steps

    def get_trajectory(self):
        return np.array(self.trajectory)

    def get_connections(self):
        return np.array([np.concatenate([su, du]) for su, du in self.connections])