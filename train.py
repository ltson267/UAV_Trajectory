import matplotlib.pyplot as plt
from config import *
from uav_intercept_env import UAVInterceptEnv
from dql_agent import DQNAgent
from greedy_agent import GreedyAgent
import numpy as np

def plot_trajectory(env, title):
    traj = env.get_trajectory()
    connections = env.get_connections()
    su_nodes = np.array(env.su_nodes)
    du_nodes = np.array(env.du_nodes)
    plt.figure(figsize=(6, 6))
    for idx, conn in enumerate(connections):
        x = [conn[0], conn[2]]
        y = [conn[1], conn[3]]
        if idx == 0:
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.7, label='SU-DU', zorder=1)
        else:
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.7, zorder=1)
    plt.scatter(su_nodes[:, 0], su_nodes[:, 1], c='orange', s=120, marker='s', edgecolors='black', label='SU', zorder=3)
    plt.scatter(du_nodes[:, 0], du_nodes[:, 1], c='purple', s=120, marker='o', edgecolors='black', label='DU', zorder=3)
    plt.plot(traj[:, 0], traj[:, 1], c='red', marker='x', label='UAV path', zorder=2)
    plt.scatter([traj[0, 0]], [traj[0, 1]], c='green', s=150, label='Start', zorder=4)
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], c='black', s=150, label='End', zorder=4)
    plt.title(f"UAV Trajectory - {title}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_agent(agent_class, episodes=EPISODES):
    env = UAVInterceptEnv()
    state = env.reset()
    agent = agent_class(state_dim=len(state), action_dim=len(ACTIONS))

    # Cập nhật thông tin kết nối cho GreedyAgent
    if hasattr(agent, 'update_connections'):
        connections = env.get_links()
        agent.update_connections(connections)

    rewards_history = []
    moving_avg_rewards = []
    completion_rate = []
    trajectory_lengths = []
    hover_efficiencies = []
    battery_levels = []
    performance_history = []  # Track performance to detect catastrophic forgetting
    best_reward = -float('inf')
    best_trajectory = None
    best_connections = None

    # Chỉ sử dụng epsilon cho DQN agent
    epsilon = EPSILON if hasattr(agent, 'epsilon') else 1.0
    min_epsilon = 0.2   # Higher minimum for sustained exploration
    decay = 0.999       # Very slow decay for better exploration

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        # Cập nhật thông tin kết nối cho GreedyAgent mỗi episode
        if hasattr(agent, 'update_connections'):
            connections = env.get_links()
            agent.update_connections(connections)

        # Chỉ set epsilon cho DQN agent
        if hasattr(agent, 'epsilon'):
            agent.epsilon = epsilon
        
        for step in range(env.max_steps):
            action = agent.act(state)
            if not (0 <= action < len(ACTIONS)):
                print(f"[ERROR] Action out of range: {action}, valid range: 0-{len(ACTIONS)-1}")
                continue
                
            next_state, reward, done = env.step(action)
            state_flat = np.array(state, dtype=np.float32).flatten()
            next_state_flat = np.array(next_state, dtype=np.float32).flatten()
            
            if state_flat.shape[0] != agent.state_dim or next_state_flat.shape[0] != agent.state_dim:
                print(f"[ERROR] State shape: {state_flat.shape}, Next state shape: {next_state_flat.shape}")
                continue
                
            action = max(0, min(action, agent.action_dim - 1))

            # Chỉ push vào memory và train đối với DQN agent
            if hasattr(agent, 'memory'):
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
        
        # Update best trajectory based on reward and trajectory efficiency
        current_length = env.get_trajectory_length()
        current_efficiency = env.get_hover_efficiency()

        # Score combines reward, trajectory length (shorter is better), and hover efficiency (higher is better)
        # Use actual trajectory length instead of step count for better optimization
        length_score = 100 / (current_length + 1) if current_length > 0 else 100
        current_score = total_reward + length_score + (current_efficiency * 50)

        if total_reward > best_reward or (total_reward >= best_reward * 0.9 and current_score > best_reward):
            best_reward = total_reward
            best_trajectory = env.get_trajectory().copy()
            best_connections = env.get_connections().copy()
        
        # Enhanced logging with trajectory metrics
        agent_type = "DQN" if hasattr(agent, 'memory') else "Greedy"
        if (ep + 1) % 50 == 0 or total_reward > 0:
            avg_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else np.mean(trajectory_lengths)
            avg_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else np.mean(hover_efficiencies)
            avg_battery = np.mean(battery_levels[-50:]) if len(battery_levels) >= 50 else np.mean(battery_levels)
            epsilon_str = f" | Epsilon: {epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(f"[{agent_type}] Episode {ep+1:04d} | Steps: {step+1:3d} | Reward: {total_reward:8.2f}{epsilon_str} | "
                   f"Moving Avg: {moving_avg_rewards[-1]:8.2f} | "
                   f"Completion: {completion:.2%} | Best: {best_reward:8.2f} | "
                   f"Length: {avg_length:6.1f} | Efficiency: {avg_efficiency:.2f} | Battery: {avg_battery:5.1f}")
        
        # Track performance for catastrophic forgetting detection (chỉ dành cho DQN)
        if hasattr(agent, 'memory'):
            current_performance = total_reward if completion >= 0.8 else total_reward - 100
            performance_history.append(current_performance)

        # Detect catastrophic forgetting and increase epsilon if needed
        if len(performance_history) >= 100:
            recent_avg = np.mean(performance_history[-50:])
            older_avg = np.mean(performance_history[-100:-50])

            # If performance dropped significantly, increase epsilon to recover
            if older_avg > 100 and recent_avg < older_avg * 0.5 and epsilon < 0.3:
                epsilon = min(0.3, epsilon * 2)  # Boost epsilon to recover from forgetting
                print(f"[DQN] Performance drop detected! Boosting epsilon to {epsilon:.3f} for recovery")

        # Epsilon decay chỉ dành cho DQN agent
        if hasattr(agent, 'epsilon') and epsilon > min_epsilon:
            epsilon *= decay
            epsilon = max(epsilon, min_epsilon)
            
        # Early stopping based on reward and trajectory efficiency (chỉ dành cho DQN)
        if hasattr(agent, 'memory') and len(moving_avg_rewards) >= 200 and moving_avg_rewards[-1] > 1000:
            # Also check if trajectory efficiency is good in recent episodes
            recent_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else 0
            recent_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else float('inf')

            # Early stopping if we have both good reward AND good trajectory efficiency
            if recent_efficiency > 0.8 and recent_length < 50:  # Good efficiency and reasonable length
                print(f"[DQN] Early stopping at episode {ep+1} - Good performance and trajectory efficiency achieved!")
                print(f"[DQN] Recent trajectory length: {recent_length:.1f}, Hover efficiency: {recent_efficiency:.3f}")
                break

    # Enhanced plotting for DQN
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(rewards_history, alpha=0.6, label='Episode Reward')
    ax1.plot(moving_avg_rewards, 'r-', linewidth=2, label='Moving Average (100)')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"{agent_class.__name__} Training Rewards")
    ax1.legend()
    ax1.grid(True)
    
    # Completion rate
    ax2.plot(completion_rate, 'g-', alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Completion Rate")
    ax2.set_title("Task Completion Rate")
    ax2.grid(True)
    
    # Reward distribution
    ax3.hist(rewards_history, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Reward")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Reward Distribution")
    ax3.grid(True)
    
    # Epsilon decay (chỉ dành cho DQN)
    if hasattr(agent, 'epsilon'):
        epsilons = [EPSILON * (decay ** i) for i in range(len(rewards_history))]
        ax4.plot(epsilons, 'purple', alpha=0.7)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")
        ax4.set_title("Epsilon Decay")
        ax4.grid(True)
    else:
        # Ẩn subplot epsilon cho GreedyAgent
        ax4.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics with trajectory metrics
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Total Episodes: {len(rewards_history)}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Final Moving Average: {moving_avg_rewards[-1]:.2f}")
    print(f"Average Completion Rate: {np.mean(completion_rate):.2%}")
    print(f"Positive Reward Episodes: {sum(1 for r in rewards_history if r > 0)}/{len(rewards_history)}")
    if hasattr(agent, 'epsilon'):
        print(f"Final Epsilon: {epsilon:.4f}")
    print(f"Average Trajectory Length: {np.mean(trajectory_lengths):.1f} units")
    print(f"Average Hover Efficiency: {np.mean(hover_efficiencies):.3f}")
    print(f"Best Trajectory Length: {min(trajectory_lengths):.1f} units")
    print(f"Best Hover Efficiency: {max(hover_efficiencies):.3f}")
    print(f"Trajectory Length Improvement: {(trajectory_lengths[0] if len(trajectory_lengths) > 0 else 0) - min(trajectory_lengths):.1f} units saved")
    print(f"Average Final Battery Level: {np.mean(battery_levels):.1f}/100.0")
    print(f"Battery Depletion Episodes: {sum(1 for b in battery_levels if b <= 0)}/{len(battery_levels)}")

    # Return results for comparison
    return {
        'best_reward': best_reward,
        'completion_rate': np.mean(completion_rate),
        'avg_trajectory_length': np.mean(trajectory_lengths),
        'avg_battery': np.mean(battery_levels)
    }

    def plot_best_trajectory(traj, connections, title):
        plt.figure(figsize=(6, 6))
        su_nodes = np.array(env.su_nodes)
        du_nodes = np.array(env.du_nodes)
        for conn in connections:
            x = [conn[0], conn[2]]
            y = [conn[1], conn[3]]
            plt.plot(x, y, c='blue', linestyle='--', alpha=0.5)
        plt.scatter(su_nodes[:, 0], su_nodes[:, 1], c='orange', s=80, marker='s', edgecolors='black', label='SU', zorder=3)
        plt.scatter(du_nodes[:, 0], du_nodes[:, 1], c='purple', s=80, marker='o', edgecolors='black', label='DU', zorder=3)
        plt.plot(traj[:, 0], traj[:, 1], c='red', marker='x', label='UAV path')
        plt.scatter([traj[0, 0]], [traj[0, 1]], c='green', s=100, label='Start')
        plt.scatter([traj[-1, 0]], [traj[-1, 1]], c='black', s=100, label='End')
        plt.title(f"UAV Trajectory (Best Reward & Efficiency) - {title}\nMax Reward: {best_reward:.2f}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

    if best_trajectory is not None:
        plot_best_trajectory(best_trajectory, best_connections, agent_class.__name__)
    else:
        print("No valid trajectory found.")

def compare_agents():
    """So sánh hiệu suất giữa DQN Agent và GreedyAgent"""
    print("=== Comparing DQN Agent vs GreedyAgent ===\n")

    # Train DQN Agent
    print("1. Training DQN Agent...")
    dqn_env = UAVInterceptEnv()
    dqn_results = train_agent_with_env(DQNAgent, dqn_env)

    print("\n" + "="*60)

    # Test GreedyAgent (chỉ chạy 1 episode để so sánh)
    print("2. Testing GreedyAgent...")
    greedy_env = UAVInterceptEnv()
    greedy_results, greedy_trajectory, greedy_connections = test_greedy_agent_with_trajectory(GreedyAgent, greedy_env)

    print("\n" + "="*60)
    print("=== COMPARISON RESULTS ===")
    print(f"DQN Agent     - Best Reward: {dqn_results['best_reward']:.2f}")
    print(f"GreedyAgent   - Reward: {greedy_results['reward']:.2f}")
    print(f"DQN Agent     - Completion Rate: {dqn_results['completion_rate']:.2%}")
    print(f"GreedyAgent   - Completion Rate: {greedy_results['completion_rate']:.2%}")
    print(f"DQN Agent     - Avg Trajectory Length: {dqn_results['avg_trajectory_length']:.1f}")
    print(f"GreedyAgent   - Trajectory Length: {greedy_results['trajectory_length']:.1f}")
    print(f"DQN Agent     - Avg Final Battery: {dqn_results['avg_battery']:.1f}")
    print(f"GreedyAgent   - Final Battery: {greedy_results['battery']:.1f}")

    if greedy_results['reward'] > dqn_results['best_reward']:
        print("\n🎉 GreedyAgent outperforms DQN Agent!")
    elif greedy_results['completion_rate'] == 1.0 and dqn_results['completion_rate'] < 1.0:
        print("\n⭐ GreedyAgent hoàn thành nhiệm vụ trong khi DQN Agent không hoàn thành!")
    else:
        print("\n📈 DQN Agent vẫn cho kết quả tốt hơn trong dài hạn")

    # Hiển thị trajectory của cả hai agent
    print("\n" + "="*60)
    print("=== VISUALIZATION ===")

    # DQN Agent trajectory (sử dụng trajectory tốt nhất)
    if dqn_results['best_trajectory'] is not None:
        plot_trajectory(dqn_env, f"DQN Agent (Best: {dqn_results['best_reward']:.2f})")

    # GreedyAgent trajectory
    if greedy_trajectory is not None:
        plot_trajectory(greedy_env, f"GreedyAgent (Reward: {greedy_results['reward']:.2f})")
    

def train_agent_with_env(agent_class, env):
    """Train agent và trả về kết quả kèm trajectory tốt nhất"""
    state = env.reset()
    agent = agent_class(state_dim=len(state), action_dim=len(ACTIONS))

    # Cập nhật thông tin kết nối cho GreedyAgent
    if hasattr(agent, 'update_connections'):
        connections = env.get_links()
        agent.update_connections(connections)

    rewards_history = []
    moving_avg_rewards = []
    completion_rate = []
    trajectory_lengths = []
    hover_efficiencies = []
    battery_levels = []
    performance_history = []
    best_reward = -float('inf')
    best_trajectory = None
    best_connections = None

    # Chỉ sử dụng epsilon cho DQN agent
    epsilon = EPSILON if hasattr(agent, 'epsilon') else 1.0
    min_epsilon = 0.2
    decay = 0.999

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        # Cập nhật thông tin kết nối cho GreedyAgent mỗi episode
        if hasattr(agent, 'update_connections'):
            connections = env.get_links()
            agent.update_connections(connections)

        # Chỉ set epsilon cho DQN agent
        if hasattr(agent, 'epsilon'):
            agent.epsilon = epsilon

        for step in range(env.max_steps):
            action = agent.act(state)

            if not (0 <= action < len(ACTIONS)):
                print(f"[ERROR] Action out of range: {action}, valid range: 0-{len(ACTIONS)-1}")
                continue

            next_state, reward, done = env.step(action)
            state_flat = np.array(state, dtype=np.float32).flatten()
            next_state_flat = np.array(next_state, dtype=np.float32).flatten()

            if state_flat.shape[0] != agent.state_dim or next_state_flat.shape[0] != agent.state_dim:
                print(f"[ERROR] State shape: {state_flat.shape}, Next state shape: {next_state_flat.shape}")
                continue

            action = max(0, min(action, agent.action_dim - 1))

            # Chỉ push vào memory và train đối với DQN agent
            if hasattr(agent, 'memory'):
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

        # Update best trajectory
        current_length = env.get_trajectory_length()
        current_efficiency = env.get_hover_efficiency()

        length_score = 100 / (current_length + 1) if current_length > 0 else 100
        current_score = total_reward + length_score + (current_efficiency * 50)

        if total_reward > best_reward or (total_reward >= best_reward * 0.9 and current_score > best_reward):
            best_reward = total_reward
            best_trajectory = env.get_trajectory().copy()
            best_connections = env.get_connections().copy()

        # Enhanced logging
        agent_type = "DQN" if hasattr(agent, 'memory') else "Greedy"
        if (ep + 1) % 100 == 0 or total_reward > 0:
            avg_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else np.mean(trajectory_lengths)
            avg_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else np.mean(hover_efficiencies)
            avg_battery = np.mean(battery_levels[-50:]) if len(battery_levels) >= 50 else np.mean(battery_levels)
            epsilon_str = f" | Epsilon: {epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(f"[{agent_type}] Episode {ep+1:04d} | Steps: {step+1:3d} | Reward: {total_reward:8.2f}{epsilon_str} | "
                   f"Moving Avg: {moving_avg_rewards[-1]:8.2f} | "
                   f"Completion: {completion:.2%} | Best: {best_reward:8.2f} | "
                   f"Length: {avg_length:6.1f} | Efficiency: {avg_efficiency:.2f} | Battery: {avg_battery:5.1f}")

        # Track performance for catastrophic forgetting detection (chỉ dành cho DQN)
        if hasattr(agent, 'memory'):
            current_performance = total_reward if completion >= 0.8 else total_reward - 100
            performance_history.append(current_performance)

            # Detect catastrophic forgetting and increase epsilon if needed
            if len(performance_history) >= 100:
                recent_avg = np.mean(performance_history[-50:])
                older_avg = np.mean(performance_history[-100:-50])

                if older_avg > 100 and recent_avg < older_avg * 0.5 and epsilon < 0.3:
                    epsilon = min(0.3, epsilon * 2)
                    print(f"[DQN] Performance drop detected! Boosting epsilon to {epsilon:.3f} for recovery")

        # Epsilon decay chỉ dành cho DQN agent
        if hasattr(agent, 'epsilon') and epsilon > min_epsilon:
            epsilon *= decay
            epsilon = max(epsilon, min_epsilon)

        # Early stopping based on reward and trajectory efficiency (chỉ dành cho DQN)
        if hasattr(agent, 'memory') and len(moving_avg_rewards) >= 200 and moving_avg_rewards[-1] > 1000:
            recent_efficiency = np.mean(hover_efficiencies[-50:]) if len(hover_efficiencies) >= 50 else 0
            recent_length = np.mean(trajectory_lengths[-50:]) if len(trajectory_lengths) >= 50 else float('inf')

            if recent_efficiency > 0.8 and recent_length < 50:
                print(f"[DQN] Early stopping at episode {ep+1} - Good performance and trajectory efficiency achieved!")
                print(f"[DQN] Recent trajectory length: {recent_length:.1f}, Hover efficiency: {recent_efficiency:.3f}")
                break

    # Return results with trajectory
    return {
        'best_reward': best_reward,
        'completion_rate': np.mean(completion_rate),
        'avg_trajectory_length': np.mean(trajectory_lengths),
        'avg_battery': np.mean(battery_levels),
        'best_trajectory': best_trajectory,
        'best_connections': best_connections
    }

def test_greedy_agent_single(agent_class):
    """Test GreedyAgent trong 1 episode để so sánh"""
    env = UAVInterceptEnv()
    state = env.reset()
    agent = agent_class(state_dim=len(state), action_dim=len(ACTIONS))

    # Cập nhật thông tin kết nối cho GreedyAgent
    if hasattr(agent, 'update_connections'):
        connections = env.get_links()
        agent.update_connections(connections)

    total_reward = 0

    for step in range(env.max_steps):
        action = agent.act(state)

        if not (0 <= action < len(ACTIONS)):
            print(f"[ERROR] Action out of range: {action}, valid range: 0-{len(ACTIONS)-1}")
            continue

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break

    completion_rate = np.sum(state[3:]) / len(state[3:])
    trajectory_length = env.get_trajectory_length()
    final_battery = state[2]

    return {
        'reward': total_reward,
        'completion_rate': completion_rate,
        'trajectory_length': trajectory_length,
        'battery': final_battery
    }

def test_greedy_agent_with_trajectory(agent_class, env):
    """Test GreedyAgent và trả về trajectory"""
    state = env.reset()
    agent = agent_class(state_dim=len(state), action_dim=len(ACTIONS))

    # Cập nhật thông tin kết nối cho GreedyAgent
    if hasattr(agent, 'update_connections'):
        connections = env.get_links()
        agent.update_connections(connections)

    total_reward = 0
    trajectory = None
    final_connections = None

    for step in range(env.max_steps):
        action = agent.act(state)

        if not (0 <= action < len(ACTIONS)):
            print(f"[ERROR] Action out of range: {action}, valid range: 0-{len(ACTIONS)-1}")
            continue

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break

    completion_rate = np.sum(state[3:]) / len(state[3:])
    trajectory_length = env.get_trajectory_length()
    final_battery = state[2]

    # Lưu trajectory cuối cùng
    trajectory = env.get_trajectory().copy()
    final_connections = env.get_connections().copy()

    return {
        'reward': total_reward,
        'completion_rate': completion_rate,
        'trajectory_length': trajectory_length,
        'battery': final_battery
    }, trajectory, final_connections

if __name__ == "__main__":
    # Chạy so sánh giữa hai agent
    compare_agents()





