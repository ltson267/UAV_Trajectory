import numpy as np

class SignalModel:
    """
    Mô hình truyền tải cho UAV signal collection dựa trên SINR và thông lượng Shannon
    """

    def __init__(self):
        # Tham số thực tế hơn cho wireless comm
        self.tx_power_dbm = 20.0          # 0.1W = 20 dBm (realistic cho UAV)
        self.frequency = 2400e6           # 2.4 GHz
        self.bandwidth = 20e6             # 20 MHz
        self.noise_power_dbm = -100.0     # Thermal noise + ambient
        self.path_loss_exp = 3.5          # Path loss exponent cho dense urban
        self.ref_distance = 1.0           # Reference distance (m)
        self.ref_path_loss = 40.0         # Path loss at 1m, 2.4GHz (dB)
        self.shadowing_std = 6.0          # Realistic shadowing ±6dB

        # Ngưỡng thu thập thực tế
        self.collection_threshold_mbps = 25.0  # Mbps (yêu cầu chất lượng tốt)

        # Reward scaling điều chỉnh cho threshold cao hơn
        self.throughput_improvement_scale = 1.0    # Reward per Mbps improvement
        self.throughput_decrease_penalty = 0.5     # Penalty per Mbps decrease
        self.min_throughput_delta = 5.0            # Threshold cho penalty

    def calculate_distance_to_segment(self, point, segment_start, segment_end):
        """
        Tính khoảng cách từ điểm đến đoạn thẳng SU-DU

        Args:
            point: np.array([x, y])
            segment_start: np.array([x, y])
            segment_end: np.array([x, y])

        Returns:
            float: khoảng cách nhỏ nhất đến segment
        """
        # Vector từ start đến end
        segment_vec = np.array(segment_end) - np.array(segment_start)
        # Vector từ start đến point
        point_vec = np.array(point) - np.array(segment_start)

        # Projection scalar
        segment_len = np.linalg.norm(segment_vec)
        if segment_len == 0:
            return np.linalg.norm(point - segment_start)

        # Tính projection
        t = np.dot(point_vec, segment_vec) / (segment_len ** 2)
        t = np.clip(t, 0, 1)

        # Điểm gần nhất trên segment
        closest_point = segment_start + t * segment_vec
        return np.linalg.norm(point - closest_point)

    def calculate_path_loss(self, distance):
        """
        Tính path loss (dB)

        Args:
            distance: khoảng cách (m)

        Returns:
            float: path loss (dB)
        """
        if distance < self.ref_distance:
            distance = self.ref_distance

        path_loss = self.ref_path_loss + 10 * self.path_loss_exp * np.log10(distance / self.ref_distance)

        # Thêm shadowing (optional - có thể bật/tắt)
        # shadowing = np.random.normal(0, self.shadowing_std)
        # path_loss += shadowing

        return path_loss

    def calculate_received_power(self, distance):
        """
        Tính công suất nhận tại UAV (dBm)

        Args:
            distance: khoảng cách đến nguồn phát (m)

        Returns:
            float: công suất nhận (dBm)
        """
        path_loss = self.calculate_path_loss(distance)
        return self.tx_power_dbm - path_loss

    def calculate_sinr(self, uav_pos, target_connection_idx, all_connections):
        """
        Tính SINR cho một kết nối cụ thể tại vị trí UAV

        Args:
            uav_pos: vị trí UAV (np.array([x, y]))
            target_connection_idx: index của kết nối muốn thu thập
            all_connections: danh sách tất cả các kết nối [(su, du), ...]

        Returns:
            float: SINR in dB
        """
        target_su, target_du = all_connections[target_connection_idx]

        # Khoảng cách từ UAV đến segment của target connection
        target_distance = self.calculate_distance_to_segment(uav_pos, target_su, target_du)

        # Công suất tín hiệu mong muốn
        signal_power_dbm = self.calculate_received_power(target_distance)
        signal_power_linear = 10**(signal_power_dbm/10)  # Chuyển sang linear

        # Công suất nhiễu và can nhiễu
        interference_power_linear = 0.0

        for i, (su, du) in enumerate(all_connections):
            if i == target_connection_idx:
                continue  # Bỏ qua tín hiệu mong muốn

            # Khoảng cách từ UAV đến interfering connection
            interf_distance = self.calculate_distance_to_segment(uav_pos, su, du)
            interf_power_dbm = self.calculate_received_power(interf_distance)
            interf_power_linear = 10**(interf_power_dbm/10)

            interference_power_linear += interf_power_linear

        # Nhiễu nền
        noise_power_linear = 10**(self.noise_power_dbm/10)

        # Tổng nhiễu
        total_noise_linear = noise_power_linear + interference_power_linear

        # Tránh chia cho 0
        if total_noise_linear == 0:
            return 100.0  # Very high SINR

        # SINR
        sinr_linear = signal_power_linear / total_noise_linear
        sinr_db = 10 * np.log10(sinr_linear)

        return sinr_db

    def calculate_throughput(self, sinr_db):
        """
        Tính thông lượng Shannon (Mbps)

        Args:
            sinr_db: SINR in dB

        Returns:
            float: throughput in Mbps
        """
        if sinr_db < 0:
            sinr_db = 0  # Tránh giá trị âm

        sinr_linear = 10**(sinr_db/10)
        throughput_bps = self.bandwidth * np.log2(1 + sinr_linear)
        throughput_mbps = throughput_bps / 1e6

        return throughput_mbps

    def can_collect_signal(self, throughput_mbps):
        """
        Kiểm tra xem có thể thu thập tín hiệu không

        Args:
            throughput_mbps: thông lượng tính được (Mbps)

        Returns:
            bool: True nếu có thể thu thập
        """
        return throughput_mbps >= self.collection_threshold_mbps

    def get_throughput_at_position(self, uav_pos, connection_idx, all_connections):
        """
        Tính thông lượng tại vị trí UAV cho một kết nối

        Args:
            uav_pos: vị trí UAV
            connection_idx: index kết nối
            all_connections: tất cả kết nối

        Returns:
            float: throughput in Mbps
        """
        sinr_db = self.calculate_sinr(uav_pos, connection_idx, all_connections)
        return self.calculate_throughput(sinr_db)

    def calculate_reward_for_movement(self, old_pos, new_pos, connection_idx, all_connections):
        """
        Tính reward cho việc di chuyển dựa trên throughput improvement

        Args:
            old_pos, new_pos: vị trí cũ và mới
            connection_idx: kết nối đang quan tâm
            all_connections: tất cả kết nối

        Returns:
            float: reward cho movement
        """
        old_throughput = self.get_throughput_at_position(old_pos, connection_idx, all_connections)
        new_throughput = self.get_throughput_at_position(new_pos, connection_idx, all_connections)

        improvement = new_throughput - old_throughput

        reward = 0.0

        if improvement > 0:
            # Reward for throughput improvement
            reward += improvement * self.throughput_improvement_scale
        elif improvement < -self.min_throughput_delta:
            # Penalty for significant throughput decrease
            reward -= abs(improvement) * self.throughput_decrease_penalty

        return reward