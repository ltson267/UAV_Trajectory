import numpy as np
import random
from config import *
from signal_model import SignalModel

class UAVInterceptEnv:
    def __init__(self, debug_components: bool = False, randomize: bool = True):
        # Respect randomize flag: avoid fixed seed when randomizing episodes
        if not randomize:
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
        # Unified parameters (throughput-centric, no mode flag)
        self.hover_distance_threshold = HOVER_DISTANCE_THRESHOLD
        self.move_cost = MOVE_BASE_COST
        self.collect_reward = COLLECT_BASE_REWARD
        self.hover_fail_penalty_near = HOVER_FAIL_PENALTY
        self.hover_fail_penalty_far = HOVER_FAIL_PENALTY * 1.2
        self.completion_reward = COMPLETION_BASE_BONUS
        self.distance_scale = 0.05  # small distance shaping to guide approach
        self.proximity_bonus = 0.0
        # Throughput tracking
        self.prev_avg_uncollected_sinr = None
        self.debug_components = debug_components
        self.randomize = randomize
        self._generate_connections(domain_randomization=randomize)
        self.reset()

    def _generate_connections(self, domain_randomization: bool = True):
        self.connections = []
        self.su_nodes = []
        self.du_nodes = []
        margin = 2
        if domain_randomization:
            num_conn = np.random.randint(3, self.num_connections + 3)
            min_len = np.random.uniform(2, 4)
            max_len = np.random.uniform(6, 10)
        else:
            num_conn = self.num_connections
            min_len = 3
            max_len = min(self.grid_x, self.grid_y) / 2

        for i in range(num_conn):
            su = np.array([
                np.random.uniform(margin, self.grid_x - margin),
                np.random.uniform(margin, self.grid_y - margin)
            ])
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
        # Episode randomization (do before initializing collected/hover_count)
        if self.randomize:
            # Regenerate map and random start
            self._generate_connections(domain_randomization=True)
            self.uav_init_pos = np.array([
                np.random.uniform(0, self.grid_x),
                np.random.uniform(0, self.grid_y)
            ])
        self.uav_pos = self.uav_init_pos.copy()
        self.battery_level = 100.0  # Adjusted for MOVE_DECAY=0.25 with 3-level battery system
        self.steps = 0
        # Initialize arrays based on current number of connections (may have changed)
        self.collected = np.zeros(len(self.connections))
        self.hover_count = np.zeros(len(self.connections))  # Track hover count per connection
        self.trajectory = [self.uav_pos.copy()]
        self.visited_positions = set()  # Track visited positions for exploration bonus
        self.last_positions = []  # Track recent positions for trajectory efficiency
        # Initialize previous target distance
        self.prev_target_distance = None
        return self.get_state()

    def get_state(self):
        # Unified state: include SINR + top-K connection features
        sinr_list = []
        distances = []
        closest_points = []
        for i, (su, du) in enumerate(self.connections):
            sinr = self.signal_model.calculate_sinr(self.uav_pos, i, self.connections)
            sinr_list.append(sinr)
            dist = self._point_to_segment_dist(self.uav_pos, su, du)
            distances.append(dist)
            closest_points.append(self._get_closest_point_on_segment(self.uav_pos, su, du))

        sinr_array = np.array(sinr_list)
        dist_array = np.array(distances)
        collected_flags = self.collected.copy()
        completion_fraction = np.mean(collected_flags)
        uncollected_mask = collected_flags == 0
        uncollected_sinr = sinr_array[uncollected_mask] if np.any(uncollected_mask) else np.array([])
        avg_uncollected_sinr = uncollected_sinr.mean() if uncollected_sinr.size > 0 else 0.0
        min_uncollected_sinr = uncollected_sinr.min() if uncollected_sinr.size > 0 else 0.0

        # Select top-K connections (priority: lowest SINR among uncollected, then distance)
        indices = list(range(len(self.connections)))
        if np.any(uncollected_mask):
            # sort by (collected first=high key to push back), then SINR ascending, then distance ascending
            sort_keys = []
            for idx in indices:
                collected = collected_flags[idx]
                sort_keys.append((collected, sinr_array[idx], dist_array[idx]))
            sorted_indices = [i for _, i in sorted(zip(sort_keys, indices), key=lambda x: (x[0][0], x[0][1], x[0][2]))]
        else:
            # all collected: just take by distance
            sorted_indices = list(np.argsort(dist_array))
        top_k = sorted_indices[:MAX_FEATURE_CONN]

        # Feature assembly per connection: [collected_flag, sinr_norm, distance_norm, dx_norm, dy_norm]
        per_conn_features = []
        for idx in top_k:
            su, du = self.connections[idx]
            cp = closest_points[idx]
            dx, dy = cp - self.uav_pos
            sinr_norm = sinr_array[idx] / SINR_SCALE
            dist_norm = dist_array[idx] / DISTANCE_SCALE
            per_conn_features.extend([
                collected_flags[idx],
                sinr_norm,
                dist_norm,
                dx / DISTANCE_SCALE,
                dy / DISTANCE_SCALE,
            ])
        # Padding if fewer than MAX_FEATURE_CONN
        while len(per_conn_features) < MAX_FEATURE_CONN * 5:
            per_conn_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Global features
        uav_x_norm = self.uav_pos[0] / GRID_SIZE[0]
        uav_y_norm = self.uav_pos[1] / GRID_SIZE[1]
        battery_norm = self.battery_level / 100.0
        avg_uncollected_norm = avg_uncollected_sinr / SINR_SCALE
        min_uncollected_norm = min_uncollected_sinr / SINR_SCALE

        state = np.array([
            uav_x_norm,
            uav_y_norm,
            battery_norm,
            completion_fraction,
            avg_uncollected_norm,
            min_uncollected_norm,
        ] + per_conn_features, dtype=np.float32)
        return state

    def _legacy_state(self):
        # Original fixed-size state (for backward compatibility / debugging)
        max_grid = 20.0
        max_battery = 100.0
        max_distance = 30.0
        max_vector = 25.0
        # Reuse earlier logic but simplified
        collected_len = len(self.connections)
        distances = []
        rel_vecs = []
        for i, (su, du) in enumerate(self.connections):
            dist = self._point_to_segment_dist(self.uav_pos, su, du)
            distances.append(dist)
            cp = self._get_closest_point_on_segment(self.uav_pos, su, du)
            dx, dy = cp - self.uav_pos
            rel_vecs.extend([dx, dy])
        best_distance = min(distances) if distances else 0.0
        state = np.concatenate([
            self.uav_pos,
            [self.battery_level],
            self.collected,
            [best_distance],
            rel_vecs
        ])
        normalized = np.zeros_like(state)
        normalized[0] = state[0] / max_grid
        normalized[1] = state[1] / max_grid
        normalized[2] = state[2] / max_battery
        normalized[3:3+collected_len] = state[3:3+collected_len]
        normalized[3+collected_len] = state[3+collected_len] / max_distance
        # Relative vectors normalization
        rv_start = 4 + collected_len
        for j in range(collected_len):
            base = rv_start + j*2
            normalized[base] = state[base] / max_vector
            normalized[base+1] = state[base+1] / max_vector
        return normalized.astype(np.float32)

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
        throughput_gain_component = 0.0
        away_penalty_component = 0.0
        hover_success_component = 0.0
        hover_fail_component = 0.0
        proximity_component = 0.0
        completion_component = 0.0
        failure_penalty_component = 0.0
        battery_penalty_component = 0.0
        collected_this_step = 0

        # Choose current nearest uncollected target and compute distance reference
        target_idx, current_target_distance = self._get_current_target()
        if self.prev_target_distance is None:
            self.prev_target_distance = current_target_distance

        # Throughput stats at start of step
        sinr_current = []
        for i in range(len(self.connections)):
            sinr_current.append(self.signal_model.calculate_sinr(self.uav_pos, i, self.connections))
        sinr_current = np.array(sinr_current)
        uncollected_mask = self.collected == 0
        avg_uncollected_before = sinr_current[uncollected_mask].mean() if np.any(uncollected_mask) else 0.0

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
            # Distance shaping towards nearest uncollected
            _, new_target_distance = self._get_current_target()
            if self.prev_target_distance is not None and new_target_distance is not None and remaining > 0 and self.distance_scale > 0.0:
                distance_delta = self.prev_target_distance - new_target_distance
                if distance_delta > 0:
                    inc = self.distance_scale * distance_delta
                    reward += inc
                    distance_component += inc
                elif distance_delta < 0:
                    dec = min(self.distance_scale * (-distance_delta), 0.2)
                    reward -= dec
                    away_penalty_component -= dec
                self.prev_target_distance = new_target_distance

            # Throughput gain shaping (uncollected only)
            sinr_after = []
            for i in range(len(self.connections)):
                sinr_after.append(self.signal_model.calculate_sinr(self.uav_pos, i, self.connections))
            sinr_after = np.array(sinr_after)
            avg_uncollected_after = sinr_after[uncollected_mask].mean() if np.any(uncollected_mask) else 0.0
            if self.prev_avg_uncollected_sinr is None:
                self.prev_avg_uncollected_sinr = avg_uncollected_before
            delta_throughput = avg_uncollected_after - self.prev_avg_uncollected_sinr
            if delta_throughput > 0:
                gain = WEIGHT_THROUGHPUT_GAIN * delta_throughput / SINR_SCALE
                reward += gain
                throughput_gain_component += gain
            elif delta_throughput < -0.5:
                penalty = THROUGHPUT_DEGRADATION_PENALTY * (-delta_throughput) / SINR_SCALE
                reward -= penalty
                away_penalty_component -= penalty
            self.prev_avg_uncollected_sinr = avg_uncollected_after
            # Removed target distance tracking in unified throughput-centric mode

            # Target-specific throughput improvement shaping (not global max)
            # Optional: throughput shaping removed to stabilize variance
            # (Retained logic could be re-enabled if needed)
                        
        elif action == "hover" and self.battery_level > 0:
            self.battery_level -= HOVER_DECAY
            collected_this_step = 0
            # Attempt to collect: choose nearest uncollected by distance
            candidate_idx = target_idx
            candidate_dist = current_target_distance if target_idx is not None else None
            if candidate_idx is not None and candidate_dist is not None and candidate_dist <= self.hover_distance_threshold:
                throughput = self.signal_model.get_throughput_at_position(self.uav_pos, candidate_idx, self.connections)
                if self.signal_model.can_collect_signal(throughput):
                    base = self.collect_reward + WEIGHT_COMPLETE
                    reward += base
                    hover_success_component += base
                    self.collected[candidate_idx] = 1
                    self.hover_count[candidate_idx] += 1
                    collected_this_step = 1
                else:
                    reward -= self.hover_fail_penalty_near
                    hover_fail_component -= self.hover_fail_penalty_near
                    self.hover_count[candidate_idx] += 1
            else:
                reward -= self.hover_fail_penalty_far
                hover_fail_component -= self.hover_fail_penalty_far
            # Update distance reference after hover
            _, newd = self._get_current_target()
            self.prev_target_distance = newd

        # Per-collection smoothing: spread former progress bonus into small residual after collection
        current_progress = np.sum(self.collected) / len(self.collected)
        # Small progressive bonus folded into collection reward: none needed here

        # Completion bonus - reward for completing all tasks
        if np.all(self.collected == 1):
            battery_efficiency = self.battery_level / 100.0
            comp = self.completion_reward + WEIGHT_BATTERY_SURPLUS * battery_efficiency
            reward += comp
            completion_component += comp
            # Throughput maintenance bonus if average SINR above target
            sinr_values = []
            for i in range(len(self.connections)):
                sinr_values.append(self.signal_model.calculate_sinr(self.uav_pos, i, self.connections))
            avg_all = np.mean(sinr_values) if sinr_values else 0.0
            if avg_all >= AVG_SINR_TARGET:
                maint = WEIGHT_THROUGHPUT_MAINTAIN * (avg_all - AVG_SINR_TARGET) / SINR_SCALE
                reward += maint
                throughput_gain_component += maint
        
        # Time penalty reduced: applied later to avoid early distortion
        # Optional time penalty removed in bounded design

        # Final failure penalty reduced
        if self.steps >= self.max_steps - 1 and not np.all(self.collected == 1):
            reward -= 2.0  # adjusted failure penalty
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
        # Replace previous reward computation with calculate_reward
        reward = self.calculate_reward(action, old_pos, collected_this_step)
        reward = np.clip(reward, -4.0, 6.0)

        done = self.is_done() or self.battery_level <= 0

        if self.debug_components:
            self.last_info = {
                'move': move_component,
                'throughput_gain': throughput_gain_component,
                'throughput_loss': away_penalty_component,
                'hover_success': hover_success_component,
                'hover_fail': hover_fail_component,
                'completion': completion_component,
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

    def _get_min_uncollected_distance(self, pos):
        dists = []
        for i,(su,du) in enumerate(self.connections):
            if self.collected[i] == 0:
                dists.append(self._point_to_segment_dist(pos, su, du))
        return min(dists) if dists else 0.0

    def calculate_reward(self, action, old_pos, collected_this_step):
        reward = 0.0
        # Level 1: Collection
        if collected_this_step:
            progress = np.sum(self.collected) / len(self.collected)
            reward += COLLECT_BASE_REWARD * (1 + progress)

        # Level 2: Potential-based shaping
        old_potential = -self._get_min_uncollected_distance(old_pos)
        new_potential = -self._get_min_uncollected_distance(self.uav_pos)
        shaping = GAMMA * new_potential - old_potential
        reward += 0.1 * shaping

        # Level 3: Sparse completion bonus
        if np.all(self.collected == 1):
            efficiency = (MAX_STEPS - self.steps) / MAX_STEPS
            battery_bonus = self.battery_level / 100.0
            reward += COMPLETION_BASE_BONUS * (1 + efficiency + battery_bonus)
        return reward

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