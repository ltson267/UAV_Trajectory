import numpy as np
import random
from config import *

class UAVInterceptEnv:
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
        self._generate_connections()
        self.reset()

    def _generate_connections(self):
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

    def get_links(self):
        """Trả về danh sách các đoạn SU-DU dạng [(su, du), ...]"""
        return [(np.array(su), np.array(du)) for su, du in self.connections]

    def reset(self):
        self.uav_pos = self.uav_init_pos.copy()
        self.battery_level = 100.0  # Adjusted for MOVE_DECAY=0.25 with 3-level battery system
        self.steps = 0
        self.collected = np.zeros(len(self.connections))
        self.trajectory = [self.uav_pos.copy()]
        self.visited_positions = set()  # Track visited positions for exploration bonus
        self.hover_count = np.zeros(len(self.connections))  # Track hover count per connection
        self.last_positions = []  # Track recent positions for trajectory efficiency
        return self.get_state()

    def get_state(self):
        state = np.concatenate([self.uav_pos, [self.battery_level], self.collected], axis=0)
        return state.astype(np.float32)

    def step(self, action_idx):
        action = self.actions[action_idx]
        previous_collected = np.sum(self.collected)
        
        # Điều chỉnh move_step theo battery (50/25/0 thresholds for battery=100)
        if self.battery_level >= 50:
            move_step = 1.0
        elif self.battery_level >= 25:
            move_step = 0.5
        else:
            move_step = 0.25

        reward = 0
        collected_this_step = 0

        # Movement và battery consumption
        if action in ["left", "right", "forward", "backward"] and self.battery_level > 0:
            old_pos = self.uav_pos.copy()
            if action == "left":
                self.uav_pos[0] = max(0, self.uav_pos[0] - move_step)
            elif action == "right":
                self.uav_pos[0] = min(self.grid_x - 1, self.uav_pos[0] + move_step)
            elif action == "forward":
                self.uav_pos[1] = min(self.grid_y - 1, self.uav_pos[1] + move_step)
            elif action == "backward":
                self.uav_pos[1] = max(0, self.uav_pos[1] - move_step)

            self.battery_level -= move_step * MOVE_DECAY

            # Track trajectory efficiency
            if len(self.last_positions) >= 5:
                self.last_positions.pop(0)
            self.last_positions.append(self.uav_pos.copy())

            # Increased movement penalty to encourage shorter paths
            reward -= 0.5
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            hover_on_collected = False

            for i, (su, du) in enumerate(self.connections):
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                if dist < 1.5 and self.collected[i] == 0:
                    # High collection reward to strongly encourage collecting
                    reward += 200
                    self.collected[i] = 1
                    self.hover_count[i] += 1
                    collected_this_step += 1
                elif dist < 1.5 and self.collected[i] == 1:
                    hover_on_collected = True
                    self.hover_count[i] += 1

            # Reduced penalty for hovering on already collected connections
            if hover_on_collected:
                reward -= 5  # Reduced penalty for hovering on collected

            # Penalty for inefficient hover (no collection)
            if collected_this_step == 0:
                reward -= 2  # Reduced penalty for hovering without collecting

        # Simplified reward structure focused on battery and trajectory efficiency

        # Increased progress bonus - encourage steady progress
        current_progress = np.sum(self.collected) / len(self.collected)
        if current_progress > previous_collected / len(self.collected):
            reward += 15 * current_progress  # Increased bonus for making progress

        # Battery efficiency bonus - reward for completing with high battery
        if np.all(self.collected == 1):
            battery_efficiency = self.battery_level / 100.0  # Fraction of battery remaining
            trajectory_length = self.get_trajectory_length()

            # Increased completion reward for better learning
            base_completion = 150

            # Battery efficiency bonus (up to 50 points for saving battery)
            battery_bonus = battery_efficiency * 50

            # Trajectory length bonus (shorter trajectory = higher bonus)
            # Optimal trajectory length is roughly 20-30, so bonus for < 40
            length_bonus = max(0, 40 - trajectory_length) * 2

            # Hover efficiency bonus (reward for efficient hovering)
            hover_efficiency = self.get_hover_efficiency()
            hover_bonus = hover_efficiency * 30

            completion_bonus = base_completion + battery_bonus + length_bonus + hover_bonus
            reward += completion_bonus
        
        # Reduced penalties to prevent agent collapse
        if self.steps >= self.max_steps - 1 and not np.all(self.collected == 1):
            reward -= 20  # Reduced penalty for not completing in time

        # Penalty for battery depletion - only if not completed
        if self.battery_level <= 0 and not np.all(self.collected == 1):
            reward -= 10  # Reduced penalty for running out of battery
            self.battery_level = 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())

        # Debug: Check if agent is stuck or not learning
        completed_all = np.all(self.collected == 1)
        if self.steps > 100 and not completed_all:
            # Agent taking too long, give hint
            reward -= 1  # Small penalty for taking too long

        done = self.is_done() or self.battery_level <= 0
        return self.get_state(), reward, done

    def _calculate_trajectory_efficiency(self):
        """Calculate battery efficiency for recent trajectory"""
        if len(self.last_positions) < 3:
            return 0.0

        # For now, return simple efficiency based on battery level
        # This encourages completing with high battery
        return self.battery_level / 100.0

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

    def get_trajectory_length(self):
        """Calculate total length of trajectory"""
        if len(self.trajectory) < 2:
            return 0.0
        total_length = 0
        for i in range(1, len(self.trajectory)):
            total_length += np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
        return total_length

    def get_hover_efficiency(self):
        """Calculate hover efficiency (optimal is 1 hover per connection)"""
        if len(self.connections) == 0:
            return 1.0

        # Perfect efficiency: 1 hover per connection
        # Efficiency decreases as hover count increases beyond 1
        total_excess_hovers = sum(max(0, count - 1) for count in self.hover_count)
        max_possible_excess = len(self.connections) * 2  # Assume max 3 hovers per connection

        if max_possible_excess == 0:
            return 1.0

        # Return efficiency as fraction of excess hovers (cap at 0 to avoid negative values)
        efficiency = 1.0 - (total_excess_hovers / max_possible_excess)
        return max(0.0, efficiency)  # Ensure non-negative efficiency

    def get_connections(self):
        return np.array([np.concatenate([su, du]) for su, du in self.connections])