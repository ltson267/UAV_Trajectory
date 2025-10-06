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
        old_pos = self.uav_pos.copy()
        
        # Simplified movement - constant step size
        move_step = 1.0

        reward = 0
        collected_this_step = 0

        # Movement và battery consumption
        if action in ["left", "right", "forward", "backward"] and self.battery_level > 0:
            if action == "left":
                self.uav_pos[0] = max(0, self.uav_pos[0] - move_step)
            elif action == "right":
                self.uav_pos[0] = min(self.grid_x - 1, self.uav_pos[0] + move_step)
            elif action == "forward":
                self.uav_pos[1] = min(self.grid_y - 1, self.uav_pos[1] + move_step)
            elif action == "backward":
                self.uav_pos[1] = max(0, self.uav_pos[1] - move_step)

            self.battery_level -= MOVE_DECAY

            # Track trajectory efficiency
            if len(self.last_positions) >= 5:
                self.last_positions.pop(0)
            self.last_positions.append(self.uav_pos.copy())

            # Movement penalty to encourage efficiency
            reward -= 0.1
            
            # Distance-based reward shaping - reward for moving closer to uncollected connections
            closest_uncollected_dist_before = float('inf')
            closest_uncollected_dist_after = float('inf')
            
            for i, (su, du) in enumerate(self.connections):
                if self.collected[i] == 0:
                    # Distance to the midpoint of the connection
                    midpoint = (su + du) / 2
                    dist_before = np.linalg.norm(old_pos - midpoint)
                    dist_after = np.linalg.norm(self.uav_pos - midpoint)
                    
                    closest_uncollected_dist_before = min(closest_uncollected_dist_before, dist_before)
                    closest_uncollected_dist_after = min(closest_uncollected_dist_after, dist_after)
            
            # Reward shaping: encourage moving closer to targets
            if closest_uncollected_dist_after < closest_uncollected_dist_before:
                improvement = closest_uncollected_dist_before - closest_uncollected_dist_after
                reward += improvement * 0.5  # Reward for getting closer
            elif closest_uncollected_dist_after > closest_uncollected_dist_before:
                worsening = closest_uncollected_dist_after - closest_uncollected_dist_before
                reward -= worsening * 0.3  # Penalty for moving away
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            hover_on_collected = False
            hover_near_target = False

            for i, (su, du) in enumerate(self.connections):
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                
                if dist < 1.5:
                    hover_near_target = True
                    if self.collected[i] == 0:
                        # Collection reward - keep it significant but balanced
                        reward += 100
                        self.collected[i] = 1
                        self.hover_count[i] += 1
                        collected_this_step += 1
                    else:
                        hover_on_collected = True
                        self.hover_count[i] += 1

            # Strong penalty for hovering on already collected connections
            if hover_on_collected:
                reward -= 15

            # Penalty for hovering not near any target
            if not hover_near_target:
                reward -= 10

        # Progress reward - scaled by number of collections
        current_progress = np.sum(self.collected) / len(self.collected)
        if current_progress > previous_collected / len(self.collected):
            # Progressive bonus that increases with more collections
            progress_bonus = 20 + (current_progress * 30)
            reward += progress_bonus

        # Completion bonus - reward for completing all tasks
        if np.all(self.collected == 1):
            battery_efficiency = self.battery_level / 100.0
            trajectory_length = self.get_trajectory_length()
            hover_efficiency = self.get_hover_efficiency()

            # Base completion reward
            base_completion = 200

            # Battery efficiency bonus (up to 100 points for high battery)
            battery_bonus = battery_efficiency * 100

            # Trajectory efficiency bonus (reward shorter paths)
            # Ideal path is around 25-35 units for 5 connections
            optimal_length = 30.0
            if trajectory_length < optimal_length:
                length_bonus = (optimal_length - trajectory_length) * 3
            else:
                length_bonus = max(0, (optimal_length - (trajectory_length - optimal_length) * 0.5))

            # Hover efficiency bonus (reward efficient hovering)
            hover_bonus = hover_efficiency * 50

            completion_bonus = base_completion + battery_bonus + length_bonus + hover_bonus
            reward += completion_bonus
        
        # Time penalty - encourage faster completion
        if self.steps > 50 and not np.all(self.collected == 1):
            # Progressive time penalty
            time_penalty = (self.steps - 50) * 0.05
            reward -= min(time_penalty, 10)  # Cap at -10

        # Strong penalty for not completing
        if self.steps >= self.max_steps - 1 and not np.all(self.collected == 1):
            reward -= 50

        # Penalty for battery depletion
        if self.battery_level <= 0:
            if not np.all(self.collected == 1):
                reward -= 30
            self.battery_level = 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())

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