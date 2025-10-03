"""
Script để test GreedyAgent độc lập với visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *
from uav_intercept_env import UAVInterceptEnv
from greedy_agent import GreedyAgent

def test_greedy_agent():
    """Test GreedyAgent với visualization đầy đủ"""
    print("=== Testing GreedyAgent ===")

    # Khởi tạo môi trường và agent
    env = UAVInterceptEnv()
    state = env.reset()

    # Khởi tạo agent với cùng kích thước state và action như DQN
    agent = GreedyAgent(state_dim=len(state), action_dim=len(ACTIONS))

    # Cập nhật thông tin kết nối cho agent
    connections = env.get_links()
    agent.update_connections(connections)

    print(f"Initial state: {state}")
    print(f"Connections: {len(connections)}")
    print(f"Grid size: {env.grid_x}x{env.grid_y}")
    print(f"UAV initial position: [{state[0]:.2f}, {state[1]:.2f}]")
    print(f"Battery level: {state[2]:.1f}")
    print(f"Uncollected connections: {np.sum(state[3:]) == 0}")

    total_reward = 0
    steps = 0
    max_steps = MAX_STEPS

    print("\n--- Running simulation ---")

    for step in range(max_steps):
        # Agent chọn action
        action_idx = agent.act(state)
        action = ACTIONS[action_idx]

        # Thực hiện action
        next_state, reward, done = env.step(action_idx)

        total_reward += reward
        steps += 1

        # In thông tin mỗi 50 bước
        if step % 50 == 0:
            uav_pos = [next_state[0], next_state[1]]
            battery = next_state[2]
            collected = np.sum(next_state[3:])
            print(f"Step {step:3d}: Action={action:8s}, Pos=[{uav_pos[0]:.1f},{uav_pos[1]:.1f}], "
                  f"Battery={battery:5.1f}, Collected={collected}/{len(connections)}, Reward={reward:6.2f}")

        state = next_state

        if done:
            print(f"\nEpisode finished at step {step+1}")
            break

    # In kết quả cuối cùng
    final_collected = np.sum(state[3:])
    completion_rate = final_collected / len(connections)
    trajectory_length = env.get_trajectory_length()

    print("\n=== Final Results ===")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Completion rate: {completion_rate:.2%}")
    print(f"Final battery: {state[2]:.1f}")
    print(f"Trajectory length: {trajectory_length:.2f}")

    if completion_rate == 1.0:
        print("🎉 Mission completed successfully!")
    else:
        print("❌ Mission failed - not all connections collected")

    # Visualization
    print("\n=== Generating Visualization ===")

    # Plot trajectory
    traj = env.get_trajectory()
    connections = env.get_connections()
    su_nodes = np.array(env.su_nodes)
    du_nodes = np.array(env.du_nodes)

    plt.figure(figsize=(10, 10))

    # Plot connections (SU-DU)
    for idx, conn in enumerate(connections):
        x = [conn[0], conn[2]]
        y = [conn[1], conn[3]]
        if idx == 0:
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.7, label='SU-DU', linewidth=2)
        else:
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.7, linewidth=2)

    # Plot SU and DU nodes
    plt.scatter(su_nodes[:, 0], su_nodes[:, 1], c='orange', s=150, marker='s',
                edgecolors='black', linewidth=2, label='SU', zorder=3)
    plt.scatter(du_nodes[:, 0], du_nodes[:, 1], c='purple', s=150, marker='o',
                edgecolors='black', linewidth=2, label='DU', zorder=3)

    # Plot UAV trajectory
    plt.plot(traj[:, 0], traj[:, 1], c='red', marker='x', markersize=8,
             linewidth=3, label='UAV path', zorder=2)

    # Plot start and end points
    plt.scatter([traj[0, 0]], [traj[0, 1]], c='green', s=200,
                edgecolors='black', linewidth=3, label='Start', zorder=4)
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], c='black', s=200,
                edgecolors='white', linewidth=3, label='End', zorder=4)

    plt.title(f"GreedyAgent Trajectory\nReward: {total_reward:.2f}, "
              f"Steps: {steps}, Battery: {state[2]:.1f}", fontsize=14, fontweight='bold')
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set axis limits with some padding
    all_x = np.concatenate([su_nodes[:, 0], du_nodes[:, 0], traj[:, 0]])
    all_y = np.concatenate([su_nodes[:, 1], du_nodes[:, 1], traj[:, 1]])
    x_margin = (max(all_x) - min(all_x)) * 0.1
    y_margin = (max(all_y) - min(all_y)) * 0.1
    plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    plt.ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    plt.tight_layout()
    plt.show()

    # Plot battery and progress over time
    plt.figure(figsize=(12, 4))

    # Battery subplot
    plt.subplot(1, 2, 1)
    battery_history = [100.0]  # Initial battery
    current_battery = 100.0

    temp_env = UAVInterceptEnv()  # For battery calculation reference
    temp_env.reset()

    for i in range(steps):
        # Simulate battery consumption (simplified)
        if i > 0:
            action_type = "hover" if traj[i][0] == traj[i-1][0] and traj[i][1] == traj[i-1][1] else "move"
            if action_type == "hover":
                current_battery -= HOVER_DECAY
            else:
                current_battery -= MOVE_DECAY
            current_battery = max(0, current_battery)
        battery_history.append(current_battery)

    plt.plot(battery_history, 'b-', linewidth=2, marker='o', markersize=4, markevery=20)
    plt.title('Battery Level Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Battery Level')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)

    # Progress subplot
    plt.subplot(1, 2, 2)
    progress_history = [0]
    current_progress = 0

    # Calculate cumulative progress (simplified)
    for i in range(1, steps + 1):
        # Simple progress calculation based on trajectory
        progress_history.append(min(i / steps, 1.0))

    plt.plot(progress_history, 'g-', linewidth=2, marker='s', markersize=4, markevery=20)
    plt.title('Mission Progress Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Progress (0-1)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

    return total_reward, completion_rate, trajectory_length

if __name__ == "__main__":
    test_greedy_agent()