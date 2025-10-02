import numpy as np
import random
from config import *

class UAVInterceptEnvPPO:
    def __init__(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.grid_x, self.grid_y = GRID_SIZE
        self.actions = ACTIONS
        self.max_steps = MAX_STEPS
        self.num_connections = NUM_CONNECTIONS
        self.uav_init_pos = np.array([
            np.random.uniform(0, self.grid_x),
            np.random.uniform(0, self.grid_y)
        ])
        self.connections = []
        self.su_nodes = []
        self.du_nodes = []
        margin = 2
        min_len = 3
        max_len = min(self.grid_x, self.grid_y) / 2
        su_x = np.linspace(margin, self.grid_x - margin, self.num_connections)
        for i in range(self.num_connections):
            su_y = np.random.uniform(margin, self.grid_y - margin)
            su = np.array([su_x[i], su_y])
            for _ in range(100):
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.uniform(min_len, max_len)
                du = su + np.array([np.cos(angle), np.sin(angle)]) * length
                du[0] = np.clip(du[0], 0, self.grid_x)
                du[1] = np.clip(du[1], 0, self.grid_y)
                if all(np.linalg.norm(du - other_du) > 2 for other_du in self.du_nodes) and all(np.linalg.norm(su - other_su) > 2 for other_su in self.su_nodes):
                    break
            self.su_nodes.append(su)
            self.du_nodes.append(du)
            self.connections.append((su, du))
        self.reset()

    def get_links(self):
        return [(np.array(su), np.array(du)) for su, du in self.connections]
    
    def reset(self):
        self.uav_pos = self.uav_init_pos.copy()
        self.battery_level = 40.0
        self.steps = 0
        self.collected = np.zeros(len(self.connections))
        self.trajectory = [self.uav_pos.copy()]
        return self.get_state()

    def get_state(self):
        state = np.concatenate([self.uav_pos, [self.battery_level], self.collected], axis=0)
        return state.astype(np.float32)

    def step(self, action_idx):
        action = self.actions[action_idx]
        if self.battery_level >= 25:
            move_step = 1.0
        elif self.battery_level >= 10:
            move_step = 0.5
        else:
            move_step = 0.2
        reward = 0
        collected_this_step = 0
        if self.battery_level <= 0 and action != "hover":
            reward -= 1.0
        near_any = False
        if action in ["left", "right", "forward", "backward"] and self.battery_level > 0:
            if action == "left":
                self.uav_pos[0] = max(0, self.uav_pos[0] - move_step)
            elif action == "right":
                self.uav_pos[0] = min(self.grid_x - 1, self.uav_pos[0] + move_step)
            elif action == "forward":
                self.uav_pos[1] = min(self.grid_y - 1, self.uav_pos[1] + move_step)
            elif action == "backward":
                self.uav_pos[1] = max(0, self.uav_pos[1] - move_step)
            self.battery_level -= move_step * MOVE_DECAY
            reward -= 0.02
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            for i, (su, du) in enumerate(self.connections):
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                if dist < 0.5 and self.collected[i] == 0:
                    reward += 30
                    self.collected[i] = 1
                    collected_this_step += 1
                elif dist < 1.5 and self.collected[i] == 0:
                    near_any = True
            if collected_this_step == 0:
                reward -= 0.05
        # Thêm reward nhỏ nếu UAV gần SU-DU mà chưa thu thập
        if near_any:
            reward += 0.5
        reward -= 0.02  
        if collected_this_step > 1:
            reward += 10 * (collected_this_step - 1)
        if np.all(self.collected == 1):
            reward += 120  
        if self.steps >= self.max_steps and not np.all(self.collected == 1):
            reward -= 5  
        if self.battery_level < 0:
            self.battery_level = 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())
        done = self.is_done() or self.battery_level <= 0
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
