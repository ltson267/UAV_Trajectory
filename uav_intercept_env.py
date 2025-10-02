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

            # Exploration bonus for visiting new positions
            pos_key = (round(self.uav_pos[0]), round(self.uav_pos[1]))
            if pos_key not in self.visited_positions:
                self.visited_positions.add(pos_key)
                reward += 0.1  # Small bonus for exploration

            # Enhanced reward for efficient trajectory
            trajectory_efficiency_bonus = self._calculate_trajectory_efficiency()
            reward += trajectory_efficiency_bonus * 0.5

            # Reward for moving closer to uncollected targets with priority to nearest target
            uncollected_connections = [(i, conn) for i, conn in enumerate(self.connections) if self.collected[i] == 0]
            if uncollected_connections:
                # Find the nearest uncollected target
                nearest_idx, nearest_dist = min(
                    [(i, self._point_to_segment_dist(self.uav_pos, su, du)) for i, (su, du) in uncollected_connections],
                    key=lambda x: x[1]
                )

                old_dist = self._point_to_segment_dist(old_pos, *self.connections[nearest_idx])
                new_dist = nearest_dist

                if new_dist < old_dist:
                    # Scale reward by distance improvement and prioritize nearest target
                    improvement = old_dist - new_dist
                    # Higher reward for moving toward nearest target
                    reward += 2.0 * improvement * (1.0 + 2.0 / (new_dist + 1.0))
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            hover_on_collected = False
            efficient_hover = False

            for i, (su, du) in enumerate(self.connections):
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                if dist < 1.5 and self.collected[i] == 0:
                    # Check if this is an efficient hover (first time or after collecting others)
                    if self.hover_count[i] == 0 or collected_this_step > 0:
                        reward += 50  # Reward for collection
                        efficient_hover = True
                    else:
                        reward += 30  # Reduced reward for repeated hovering

                    self.collected[i] = 1
                    self.hover_count[i] += 1
                    collected_this_step += 1
                elif dist < 1.5 and self.collected[i] == 1:
                    hover_on_collected = True
                    self.hover_count[i] += 1

            # Penalty for hovering on already collected connections (increased based on hover count)
            if hover_on_collected:
                # Get the maximum hover count among collected connections being hovered
                max_hover_on_collected = max([self.hover_count[i] for i, (_, _) in enumerate(self.connections)
                                            if self.collected[i] == 1 and self._point_to_segment_dist(self.uav_pos, *self.connections[i]) < 1.5], default=0)
                reward -= 5.0 * (1 + max_hover_on_collected)  # Increased penalty based on repeated hovering

            # Penalty for inefficient hover (no collection and not on collected connection)
            if collected_this_step == 0 and not hover_on_collected:
                # Check if UAV is close to any uncollected connection but not collecting
                close_to_uncollected = any(self._point_to_segment_dist(self.uav_pos, su, du) < 2.0
                                         for i, (su, du) in enumerate(self.connections) if self.collected[i] == 0)
                if close_to_uncollected:
                    reward -= 3.0  # Penalty for inefficient hover when close to target
                else:
                    reward -= 1.0  # Small penalty for hovering far from targets

        # Optimized reward structure for trajectory efficiency
        # Small time penalty
        reward -= 0.1

        # Reduced bonus for multiple collections in one step (encourage sequential collection)
        if collected_this_step > 1:
            reward += 10 * (collected_this_step - 1)  # Further reduced bonus

        # Progress bonus with emphasis on trajectory efficiency
        current_progress = np.sum(self.collected) / len(self.collected)
        if current_progress > previous_collected / len(self.collected):
            # Base progress bonus
            base_bonus = 3 * current_progress
            # Additional bonus for efficient trajectory
            trajectory_bonus = self._calculate_trajectory_efficiency() * 5
            reward += base_bonus + trajectory_bonus

        # Enhanced completion bonus based on trajectory efficiency
        if np.all(self.collected == 1):
            # Calculate trajectory efficiency for the entire path
            total_efficiency = self._calculate_trajectory_efficiency()
            # Bonus based on actual trajectory length (not just steps) and efficiency
            actual_trajectory_length = self.get_trajectory_length()
            # Normalize length bonus: shorter trajectories get higher bonus
            length_bonus = max(0, 100 - actual_trajectory_length)  # Up to 100 bonus for short trajectories
            efficiency_bonus = total_efficiency * 50  # Up to 50 bonus points for efficiency
            completion_bonus = 150 + length_bonus + efficiency_bonus  # Base 150 + length + efficiency bonus
            reward += completion_bonus
        
        # Reduced penalty for timeout
        if self.steps >= self.max_steps - 1 and not np.all(self.collected == 1):
            reward -= 50  # Increased penalty

        # Penalty for battery depletion
        if self.battery_level <= 0:
            reward -= 30  # Increased penalty
            self.battery_level = 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())
        done = self.is_done() or self.battery_level <= 0
        return self.get_state(), reward, done

    def _calculate_trajectory_efficiency(self):
        """Calculate trajectory efficiency bonus based on recent movement pattern"""
        if len(self.last_positions) < 3:
            return 0.0

        # Calculate total distance traveled in recent steps
        total_distance = 0
        for i in range(1, len(self.last_positions)):
            total_distance += np.linalg.norm(self.last_positions[i] - self.last_positions[i-1])

        # Calculate straight-line distance from start to end of recent positions
        start_pos = self.last_positions[0]
        end_pos = self.last_positions[-1]
        straight_distance = np.linalg.norm(end_pos - start_pos)

        # Efficiency is how much progress we made relative to distance traveled
        if total_distance > 0:
            efficiency = straight_distance / total_distance
            # Bonus for efficient movement (close to straight line)
            return min(efficiency * 0.5, 0.5)  # Cap at 0.5 bonus
        return 0.0

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
        """Calculate hover efficiency (less hovers per connection is better)"""
        if len(self.connections) == 0:
            return 1.0
        avg_hover_count = np.mean(self.hover_count)
        # Efficiency decreases as hover count increases
        return 1.0 / (1.0 + avg_hover_count)

    def get_connections(self):
        return np.array([np.concatenate([su, du]) for su, du in self.connections])