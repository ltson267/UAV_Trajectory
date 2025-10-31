"""
Script để train DQN Agent độc lập
"""
import matplotlib.pyplot as plt
from config import *
from uav_intercept_env import UAVInterceptEnv
from dql_agent import DQNAgent
import numpy as np

def train_dqn(episodes=EPISODES):
    """Train DQN Agent"""
    print("=== Training DQN Agent ===")

    env = UAVInterceptEnv()
    state = env.reset()
    agent = DQNAgent(state_dim=len(state), action_dim=len(ACTIONS))

    rewards_history = []
    moving_avg_rewards = []
    completion_rate = []
    trajectory_lengths = []
    hover_efficiencies = []
    battery_levels = []
    best_reward = -float('inf')
    best_trajectory = None
    best_connections = None

    epsilon = EPSILON
    min_epsilon = 0.1  # Lower minimum for better exploitation later
    decay = 0.997      # Slower decay for sustained exploration

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        agent.epsilon = epsilon

        for step in range(env.max_steps):
            action = agent.act(state)

            if not (0 <= action < len(ACTIONS)):
                print(f"[ERROR] Action out of range: {action}")
                continue

            next_state, reward, done = env.step(action)
            state_flat = np.array(state, dtype=np.float32).flatten()
            next_state_flat = np.array(next_state, dtype=np.float32).flatten()

            if state_flat.shape[0] != agent.state_dim or next_state_flat.shape[0] != agent.state_dim:
                print(f"[ERROR] State shape mismatch")
                continue

            action = max(0, min(action, agent.action_dim - 1))
            agent.memory.push(state_flat, action, reward, next_state_flat, int(done))
            agent.train()
            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        # Calculate moving average
        if len(rewards_history) >= 100:
            moving_avg = np.mean(rewards_history[-100:])
            moving_avg_rewards.append(moving_avg)
        else:
            moving_avg_rewards.append(np.mean(rewards_history))

        # Calculate completion rate
        completion = np.sum(env.collected) / len(env.collected)
        completion_rate.append(completion)

        # Calculate trajectory metrics
        trajectory_length = env.get_trajectory_length()
        hover_efficiency = env.get_hover_efficiency()
        final_battery = env.battery_level
        trajectory_lengths.append(trajectory_length)
        hover_efficiencies.append(hover_efficiency)
        battery_levels.append(final_battery)

        # Update best trajectory (chỉ dựa trên reward thuần túy)
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = env.get_trajectory().copy()
            best_connections = env.get_connections().copy()

        # Logging
        if (ep + 1) % 50 == 0 or total_reward > 0:
            avg_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else np.mean(trajectory_lengths)
            avg_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else np.mean(hover_efficiencies)
            avg_battery = np.mean(battery_levels[-50:]) if len(battery_levels) >= 50 else np.mean(battery_levels)
            print(f"[DQN] Episode {ep+1:04d} | Steps: {step+1:3d} | Reward: {total_reward:8.2f} | "
                   f"Epsilon: {epsilon:.3f} | Moving Avg: {moving_avg_rewards[-1]:8.2f} | "
                   f"Completion: {completion:.2%} | Best: {best_reward:8.2f} | "
                   f"Length: {avg_length:6.1f} | Efficiency: {avg_efficiency:.2f} | Battery: {avg_battery:5.1f}")

        # Epsilon decay
        if epsilon > min_epsilon:
            epsilon *= decay
            epsilon = max(epsilon, min_epsilon)

        # Early stopping - adjusted thresholds for new reward system
        if len(moving_avg_rewards) >= 200:
            recent_avg = moving_avg_rewards[-1]
            recent_completion = np.mean(completion_rate[-100:]) if len(completion_rate) >= 100 else 0
            recent_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else 0
            recent_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else float('inf')

            # Success criteria: good completion rate, good efficiency, reasonable path length
            if recent_avg > 300 and recent_completion > 0.9 and recent_efficiency > 0.75 and recent_length < 40:
                print(f"[DQN] Early stopping at episode {ep+1} - Good performance achieved!")
                print(f"       Avg Reward: {recent_avg:.2f}, Completion: {recent_completion:.2%}, "
                      f"Efficiency: {recent_efficiency:.2f}, Length: {recent_length:.1f}")
                break

    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(rewards_history, alpha=0.6, label='Episode Reward')
    ax1.plot(moving_avg_rewards, 'r-', linewidth=2, label='Moving Average (100)')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("DQN Training Rewards")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(completion_rate, 'g-', alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Completion Rate")
    ax2.set_title("Task Completion Rate")
    ax2.grid(True)

    ax3.hist(rewards_history, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Reward")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Reward Distribution")
    ax3.grid(True)

    epsilons = [EPSILON * (decay ** i) for i in range(len(rewards_history))]
    ax4.plot(epsilons, 'purple', alpha=0.7)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.set_title("Epsilon Decay")
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    # Final statistics
    print("\n=== DQN TRAINING COMPLETED ===")
    print(f"Total Episodes: {len(rewards_history)}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Final Moving Average: {moving_avg_rewards[-1]:.2f}")
    print(f"Average Completion Rate: {np.mean(completion_rate):.2%}")
    print(f"Average Trajectory Length: {np.mean(trajectory_lengths):.1f} units")
    print(f"Average Hover Efficiency: {np.mean(hover_efficiencies):.3f}")
    print(f"Average Final Battery Level: {np.mean(battery_levels):.1f}/100.0")

    # Plot best trajectory
    if best_trajectory is not None:
        plt.figure(figsize=(8, 8))
        su_nodes = np.array(env.su_nodes)
        du_nodes = np.array(env.du_nodes)

        for conn in best_connections:
            x = [conn[0], conn[2]]
            y = [conn[1], conn[3]]
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.7)

        plt.scatter(su_nodes[:, 0], su_nodes[:, 1], c='orange', s=80, marker='s', edgecolors='black', label='SU')
        plt.scatter(du_nodes[:, 0], du_nodes[:, 1], c='purple', s=80, marker='o', edgecolors='black', label='DU')
        plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], c='red', marker='x', label='UAV path')
        plt.scatter([best_trajectory[0, 0]], [best_trajectory[0, 1]], c='green', s=100, label='Start')
        plt.scatter([best_trajectory[-1, 0]], [best_trajectory[-1, 1]], c='black', s=100, label='End')

        plt.title(f"DQN Agent - Best Trajectory (Reward: {best_reward:.2f})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    train_dqn()