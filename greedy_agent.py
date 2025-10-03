import numpy as np
from config import *

class GreedyAgent:
    """
    GreedyAgent - Baseline agent sử dụng heuristic để chọn action tốt nhất

    Logic hoạt động:
    1. Nếu ở gần kết nối chưa thu thập (< 1.5 units) → hover để thu thập
    2. Ngược lại, di chuyển đến kết nối gần nhất chưa thu thập
    3. Ưu tiên khoảng cách Euclidean ngắn nhất
    4. Nếu pin thấp (< 25%), giảm hover để tiết kiệm pin
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = ACTIONS  # ["left", "right", "forward", "backward", "hover"]
        self.hover_threshold = 1.5  # Khoảng cách để có thể hover thu thập
        self.low_battery_threshold = 25.0  # Mức pin thấp
        self.connections = None  # Thông tin kết nối từ môi trường

    def act(self, state):
        """
        Chọn action tốt nhất dựa trên heuristic

        Args:
            state: numpy array chứa [uav_x, uav_y, battery_level, collected_0, collected_1, ...]

        Returns:
            action_idx: index của action được chọn
        """
        # Parse state
        uav_pos = np.array([state[0], state[1]])
        battery_level = state[2]
        collected = state[3:].astype(bool)

        # Nếu không có thông tin kết nối, hover mặc định
        if self.connections is None:
            return self.actions.index("hover")

        # Tìm kết nối gần nhất chưa thu thập
        nearest_connection = self._find_nearest_uncollected_connection(uav_pos, collected)

        if nearest_connection is None:
            # Tất cả đã thu thập xong, hover để tiết kiệm pin
            return self.actions.index("hover")

        nearest_pos = nearest_connection['pos']
        distance = nearest_connection['distance']

        # Nếu ở gần kết nối chưa thu thập và có đủ pin, hover để thu thập
        if distance < self.hover_threshold and battery_level > 10:
            return self.actions.index("hover")

        # Nếu pin thấp, ưu tiên hover ít hơn để tiết kiệm pin
        if battery_level < self.low_battery_threshold and distance > 3:
            # Ở xa và pin thấp, vẫn cố gắng di chuyển đến gần hơn
            return self._move_towards_target(uav_pos, nearest_pos, battery_level)

        # Ngược lại, di chuyển đến kết nối gần nhất
        return self._move_towards_target(uav_pos, nearest_pos, battery_level)

    def _find_nearest_uncollected_connection(self, uav_pos, collected):
        """
        Tìm kết nối gần nhất chưa thu thập

        Args:
            uav_pos: vị trí UAV hiện tại [x, y]
            collected: mảng boolean trạng thái thu thập các kết nối

        Returns:
            dict với keys: 'pos', 'distance', 'index' hoặc None nếu không tìm thấy
        """
        if self.connections is None or len(self.connections) == 0:
            return None

        min_distance = float('inf')
        nearest_connection = None

        for i, (su, du) in enumerate(self.connections):
            if collected[i]:  # Đã thu thập rồi, bỏ qua
                continue

            # Tính khoảng cách đến SU
            su_distance = np.linalg.norm(uav_pos - su)

            # Nếu UAV ở gần SU, ưu tiên SU
            if su_distance < min_distance:
                min_distance = su_distance
                nearest_connection = {
                    'pos': su,
                    'distance': su_distance,
                    'index': i
                }

            # Cũng kiểm tra DU
            du_distance = np.linalg.norm(uav_pos - du)
            if du_distance < min_distance:
                min_distance = du_distance
                nearest_connection = {
                    'pos': du,
                    'distance': du_distance,
                    'index': i
                }

        return nearest_connection

    def update_connections(self, connections):
        """
        Cập nhật thông tin kết nối từ môi trường

        Args:
            connections: danh sách kết nối dạng [(su, du), ...]
        """
        self.connections = connections

    def _move_towards_target(self, current_pos, target_pos, battery_level):
        """
        Di chuyển theo hướng đến mục tiêu

        Args:
            current_pos: vị trí hiện tại [x, y]
            target_pos: vị trí mục tiêu [x, y]
            battery_level: mức pin hiện tại

        Returns:
            action_idx: index của action di chuyển
        """
        # Tính vector hướng đến mục tiêu
        direction = target_pos - current_pos
        abs_x = abs(direction[0])
        abs_y = abs(direction[1])

        # Chọn hướng di chuyển ưu tiên theo trục có khoảng cách lớn hơn
        if abs_x > abs_y:
            # Ưu tiên di chuyển theo trục X
            if direction[0] > 0:
                return self.actions.index("right")
            else:
                return self.actions.index("left")
        else:
            # Ưu tiên di chuyển theo trục Y
            if direction[1] > 0:
                return self.actions.index("forward")
            else:
                return self.actions.index("backward")

    def train(self):
        pass