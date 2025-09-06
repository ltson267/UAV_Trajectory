import matplotlib.pyplot as plt
from config import *
from uav_intercept_env import UAVInterceptEnv
from dql_agent import DQNAgent
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
    rewards_history = []
    best_reward = -float('inf')
    best_trajectory = None

    epsilon = EPSILON
    min_epsilon = 0.02
    decay = 0.995

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        agent.epsilon = epsilon 
        for _ in range(env.max_steps):
            action = agent.act(state)
            if not (0 <= action < len(ACTIONS)):
                print(f"[ERROR] Action out of range: {action}, valid range: 0-{len(ACTIONS)-1}")
            next_state, reward, done = env.step(action)
            state_flat = np.array(state, dtype=np.float32).flatten()
            next_state_flat = np.array(next_state, dtype=np.float32).flatten()
            if state_flat.shape[0] != agent.state_dim or next_state_flat.shape[0] != agent.state_dim:
                print(f"[ERROR] State shape: {state_flat.shape}, Next state shape: {next_state_flat.shape}")
                continue
            action = max(0, min(action, agent.action_dim - 1))
            agent.memory.push(state_flat, action, reward, next_state_flat, int(done))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break
        rewards_history.append(total_reward)
        print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = env.get_trajectory().copy()
            best_connections = env.get_connections().copy()
        
        if epsilon > min_epsilon:
            epsilon *= decay
            epsilon = max(epsilon, min_epsilon)

    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{agent_class.__name__} Reward")
    plt.show()

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
        plt.title(f"UAV Trajectory (Best Reward) - {title}\nMax Reward: {best_reward:.2f}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

    if best_trajectory is not None:
        plot_best_trajectory(best_trajectory, best_connections, agent_class.__name__)
    else:
        print("No valid trajectory found.")

if __name__ == "__main__":
    print("Training DQN...")
    train_agent(DQNAgent)





