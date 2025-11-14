import numpy as np
import random
from config import *
from signal_model import SignalModel

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
        self.signal_model = SignalModel()  # Initialize signal model
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
        # Tính khoảng cách đến kết nối có SINR cao nhất chưa thu thập
        max_sinr = -float('inf')
        best_distance = float('inf')

        for i, (su, du) in enumerate(self.connections):
            if self.collected[i] == 0:  # Chỉ tính cho kết nối chưa thu thập
                sinr = self.signal_model.calculate_sinr(self.uav_pos, i, self.connections)
                if sinr > max_sinr:
                    max_sinr = sinr
                    best_distance = self._point_to_segment_dist(self.uav_pos, su, du)

        # Nếu không có kết nối nào chưa thu thập, set khoảng cách = 0
        if max_sinr == -float('inf'):
            best_distance = 0.0

        # Thêm vector tương đối đến điểm gần nhất trên các kết nối chưa thu thập
        relative_vectors = []
        for i, (su, du) in enumerate(self.connections):
            if self.collected[i] == 0:  # Chỉ cho kết nối chưa thu thập
                # Tìm điểm gần nhất trên segment
                closest_point = self._get_closest_point_on_segment(self.uav_pos, su, du)
                # Tính vector tương đối từ UAV đến điểm đó
                dx = closest_point[0] - self.uav_pos[0]
                dy = closest_point[1] - self.uav_pos[1]
                relative_vectors.extend([dx, dy])
            else:
                # Cho kết nối đã thu thập, set vector về 0
                relative_vectors.extend([0.0, 0.0])

        state = np.concatenate([self.uav_pos, [self.battery_level], self.collected, [best_distance], relative_vectors], axis=0)
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

            # Movement penalty to encourage efficiency (reduced)
            reward -= 0.01
            
            # SINR-based reward shaping - reward for moving closer to connections with high SINR
            if collected_this_step == 0:  # Only apply reward when not collecting in this step
                # Tính SINR cho tất cả kết nối chưa thu thập ở vị trí cũ và mới
                old_max_sinr = -float('inf')
                new_max_sinr = -float('inf')

                for i, (su, du) in enumerate(self.connections):
                    if self.collected[i] == 0:
                        old_sinr = self.signal_model.calculate_sinr(old_pos, i, self.connections)
                        new_sinr = self.signal_model.calculate_sinr(self.uav_pos, i, self.connections)
                        old_max_sinr = max(old_max_sinr, old_sinr)
                        new_max_sinr = max(new_max_sinr, new_sinr)

                if old_max_sinr == -float('inf'):
                    old_max_sinr = 0.0
                if new_max_sinr == -float('inf'):
                    new_max_sinr = 0.0

                # Reward cho việc tăng SINR tối đa (increased scale)
                sinr_improvement = new_max_sinr - old_max_sinr
                if sinr_improvement > 0:
                    reward += min(sinr_improvement * 0.1, 2.0)  # Scale by 0.1, cap at 2.0 per step
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            collected_this_step = 0
            hover_on_collected = False

            for i, (su, du) in enumerate(self.connections):
                if self.collected[i] == 0:  # Chỉ kiểm tra các kết nối chưa thu thập
                    throughput = self.signal_model.get_throughput_at_position(
                        self.uav_pos, i, self.connections)

                    if self.signal_model.can_collect_signal(throughput):
                        # Collection reward
                        reward += 100
                        self.collected[i] = 1
                        self.hover_count[i] += 1
                        collected_this_step += 1
                    else:
                        # Penalty cho hover không hiệu quả (reduced)
                        reward -= 1
                        self.hover_count[i] += 1
                elif self.collected[i] == 1:
                    # Penalty cho hover trên đã thu thập (reduced)
                    reward -= 3
                    self.hover_count[i] += 1

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

    def _get_closest_point_on_segment(self, p, a, b):
        """Tính điểm gần nhất trên đoạn thẳng từ điểm p đến segment a-b"""
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return closest

    def get_connections(self):
        return np.array([np.concatenate([su, du]) for su, du in self.connections])