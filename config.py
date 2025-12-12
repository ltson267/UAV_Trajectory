GRID_SIZE = (20, 20)     # Lưới
GAMMA = 0.99           
EPSILON = 1.0           # Start với max exploration
EPSILON_MIN = 0.05      
EPSILON_DECAY = 0.998   
BATCH_SIZE = 64         
LEARNING_RATE = 0.0001  
EPISODES = 1500        # More episodes với randomization
MAX_STEPS = 300         # Reduce để encourage efficiency
ACTIONS = ["left", "right", "forward", "backward", "hover"]
SEED = 42
NUM_CONNECTIONS = 5
MOVE_DECAY = 0.15       # Increased to create battery pressure
HOVER_DECAY = 0.1       # Increased hover cost     

# Max number of per-connection feature slots in state (top-K)
MAX_FEATURE_CONN = 8
# Approximate scaling constants for normalization
SINR_SCALE = 50.0          # Divide raw SINR to bring into ~0-1 range
DISTANCE_SCALE = 30.0      # Max expected segment distance for normalization

# Reward priority weights (completion > throughput > battery)
WEIGHT_COMPLETE = 3.0      # Weight added when a new connection is collected
WEIGHT_THROUGHPUT_GAIN = 0.4  # Per-step reward for improving avg SINR of uncollected
WEIGHT_THROUGHPUT_MAINTAIN = 0.2  # After full completion if avg SINR above threshold
WEIGHT_BATTERY_SURPLUS = 0.5  # Added to final completion bonus scaled by remaining battery fraction

# Throughput maintenance thresholds
AVG_SINR_TARGET = 12.0     # Target average SINR (domain dependent)
MIN_SINR_TARGET = 5.0      # Minimum SINR acceptable per link for maintenance bonus

# Hover/collection parameters (disaster mode)
HOVER_DISTANCE_THRESHOLD = 1.8  # Slightly larger tolerance
COLLECT_BASE_REWARD = 1.2       # Base reward before applying completion weight
COMPLETION_BASE_BONUS = 5.0     # Base bonus when all collected (will add battery scaled)
MOVE_STEP_SIZE = 1.0            # Movement step size grid units
MOVE_BASE_COST = 0.01           # Cost per movement action
HOVER_COST = 0.08               # Battery cost replaced/augmented if desired
HOVER_FAIL_PENALTY = 0.3        # Penalty when hover fails to collect
THROUGHPUT_DEGRADATION_PENALTY = 0.25  # Penalty when avg SINR worsens beyond tolerance

# Soft update parameter for target network (if used)
TAU = 0.005

# Curriculum learning: number of connections by episode range
CURRICULUM_SCHEDULE = {
	0: 2,      # Episodes 0-199: 2 connections
	200: 3,    # Episodes 200-399: 3 connections  
	400: 4,    # Episodes 400-599: 4 connections
	600: 5,    # Episodes 600+: 5 connections
}
