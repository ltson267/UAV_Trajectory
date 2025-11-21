import numpy as np
import random
from config import *
from signal_model import SignalModel

class UAVInterceptEnv:
    def __init__(self, debug_components: bool = False):
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
        # Track previous target distance for shaping reward
        self.prev_target_distance = None
        # Threshold distance for allowing effective hover collection
        self.hover_distance_threshold = 1.5
        # Reward scale parameters (re-tuned)
        self.move_cost = 0.01            # smaller movement cost
        self.collect_reward = 1.2         # successful collection reward (smaller for tighter bounds)
        self.hover_fail_penalty_near = 0.15
        self.hover_fail_penalty_far = 0.35
        self.completion_reward = 4.0      # terminal bonus
        self.distance_scale = 0.15        # scaling for potential-based distance improvement (reduced)
        self.proximity_bonus = 0.05       # bonus each step when within threshold but not hovering
        self.debug_components = debug_components
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
        # Initialize previous target distance
        self.prev_target_distance = None
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

        # Normalize state để cùng scale (0-1 hoặc -1 đến 1)
        state = np.concatenate([self.uav_pos, [self.battery_level], self.collected, [best_distance], relative_vectors], axis=0)

        # Normalization parameters
        max_grid = 20.0  # GRID_SIZE max
        max_battery = 100.0
        max_distance = 30.0  # Approximate max distance
        max_vector = 25.0  # Approximate max vector magnitude

        # Normalize từng phần
        normalized_state = np.zeros_like(state)

        # UAV position: 0-20 → 0-1
        normalized_state[0] = state[0] / max_grid
        normalized_state[1] = state[1] / max_grid

        # Battery: 0-100 → 0-1
        normalized_state[2] = state[2] / max_battery

        # Collected status: 0-1 (đã normalized)
        normalized_state[3:8] = state[3:8]

        # Best distance: 0-30 → 0-1
        normalized_state[8] = state[8] / max_distance

        # Relative vectors: ~-25 to +25 → ~-1 to +1
        for i in range(5):
            base_idx = 9 + i * 2
            normalized_state[base_idx] = state[base_idx] / max_vector
            normalized_state[base_idx + 1] = state[base_idx + 1] / max_vector

        return normalized_state.astype(np.float32)

    def step(self, action_idx):
        action = self.actions[action_idx]
        previous_collected = np.sum(self.collected)
        old_pos = self.uav_pos.copy()
        
        # Simplified movement - constant step size
        move_step = 1.0

        reward = 0.0
        # Component tracking for diagnostics
        move_component = 0.0
        distance_component = 0.0
        away_penalty_component = 0.0
        hover_success_component = 0.0
        hover_fail_component = 0.0
        proximity_component = 0.0
        completion_component = 0.0
        failure_penalty_component = 0.0
        battery_penalty_component = 0.0
        collected_this_step = 0

        # Determine current target (nearest uncollected connection segment)
        target_idx, current_target_distance = self._get_current_target()
        # Distance shaping: reward for reducing distance to target
        if self.prev_target_distance is None:
            self.prev_target_distance = current_target_distance

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

            # Base movement penalty
            remaining = max(1, self.num_connections - int(previous_collected))
            # Fixed movement cost regardless of remaining targets
            reward -= self.move_cost
            move_component -= self.move_cost

            # Update target distance after movement
            _, new_target_distance = self._get_current_target()
            if self.prev_target_distance is not None and new_target_distance is not None and remaining > 0:
                # Potential-based shaping: improvement only; moving away limited penalty
                distance_delta = self.prev_target_distance - new_target_distance
                if distance_delta > 0:
                    inc = self.distance_scale * distance_delta
                    reward += inc
                    distance_component += inc
                elif distance_delta < 0:
                    dec = min(self.distance_scale * (-distance_delta), 0.2)
                    reward -= dec
                    away_penalty_component -= dec

            # Proximity shaping: encourage moving into threshold before hover
            if new_target_distance is not None and new_target_distance <= self.hover_distance_threshold and action != "hover":
                reward += self.proximity_bonus
                proximity_component += self.proximity_bonus
            self.prev_target_distance = new_target_distance

            # Target-specific throughput improvement shaping (not global max)
            # Optional: throughput shaping removed to stabilize variance
            # (Retained logic could be re-enabled if needed)
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            collected_this_step = 0
            # Single target collection logic
            if target_idx is not None and self.collected[target_idx] == 0 and current_target_distance is not None:
                # Only attempt collection if within threshold; otherwise treat as inefficient hover
                if current_target_distance <= self.hover_distance_threshold:
                    throughput = self.signal_model.get_throughput_at_position(self.uav_pos, target_idx, self.connections)
                    if self.signal_model.can_collect_signal(throughput):
                        reward += self.collect_reward
                        hover_success_component += self.collect_reward
                        self.collected[target_idx] = 1
                        self.hover_count[target_idx] += 1  # count only on target
                        collected_this_step = 1
                    else:
                        # Near target but insufficient throughput
                        reward -= self.hover_fail_penalty_near
                        hover_fail_component -= self.hover_fail_penalty_near
                        self.hover_count[target_idx] += 1
                else:
                    # Hovering too far from target - stronger penalty
                    reward -= self.hover_fail_penalty_far
                    hover_fail_component -= self.hover_fail_penalty_far
                    # count still as attempt on target
                    self.hover_count[target_idx] += 1
            else:
                # Hovering with no valid target (either all collected or misaligned)
                reward -= self.hover_fail_penalty_far
                hover_fail_component -= self.hover_fail_penalty_far
                # Do not increment hover counts for already collected connections to avoid efficiency collapse

            # After hover, update target distance reference (target may change after collection)
            new_idx, new_dist = self._get_current_target()
            self.prev_target_distance = new_dist

        # Per-collection smoothing: spread former progress bonus into small residual after collection
        current_progress = np.sum(self.collected) / len(self.collected)
        # Small progressive bonus folded into collection reward: none needed here

        # Completion bonus - reward for completing all tasks
        if np.all(self.collected == 1):
            battery_efficiency = self.battery_level / 100.0
            trajectory_length = self.get_trajectory_length()
            hover_efficiency = self.get_hover_efficiency()
            comp = self.completion_reward + battery_efficiency * 2.0
            reward += comp
            completion_component += comp
        
        # Time penalty reduced: applied later to avoid early distortion
        # Optional time penalty removed in bounded design

        # Final failure penalty reduced
        if self.steps >= self.max_steps - 1 and not np.all(self.collected == 1):
            reward -= 3.0  # small failure penalty
            failure_penalty_component -= 3.0

        # Penalty for battery depletion
        if self.battery_level <= 0:
            if not np.all(self.collected == 1):
                reward -= 1.5
                battery_penalty_component -= 1.5
            self.battery_level = 0

        self.steps += 1
        self.trajectory.append(self.uav_pos.copy())

        # Clip reward to narrower band after smoothing
        # Final clip to bounded range suitable for value updates
        reward = np.clip(reward, -3.0, 3.0)

        done = self.is_done() or self.battery_level <= 0

        if self.debug_components:
            self.last_info = {
                'move': move_component,
                'distance_improve': distance_component,
                'distance_away_penalty': away_penalty_component,
                'hover_success': hover_success_component,
                'hover_fail': hover_fail_component,
                'completion': completion_component,
                'proximity': proximity_component,
                'failure_penalty': failure_penalty_component,
                'battery_penalty': battery_penalty_component,
                'raw_reward': reward
            }
        else:
            self.last_info = None

        return self.get_state(), reward, done, (self.last_info if self.debug_components else {})

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

    def _get_current_target(self):
        """Select nearest uncollected connection segment as target.
        Returns (index, distance) or (None, None) if all collected."""
        min_dist = float('inf')
        target_idx = None
        for i, (su, du) in enumerate(self.connections):
            if self.collected[i] == 0:
                dist = self._point_to_segment_dist(self.uav_pos, su, du)
                if dist < min_dist:
                    min_dist = dist
                    target_idx = i
        if target_idx is None:
            return None, None
        return target_idx, min_dist