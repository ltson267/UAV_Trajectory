import matplotlib.pyplot as plt
from config import *
from uav_env import UAVInterceptEnv
from dql_agent import DQNAgent
import numpy as np

def plot_trajectory(env, title):
    traj = env.get_trajectory()
    users = env.get_users()
    
    cluster_centers = [np.mean(np.array(cluster), axis=0) for cluster in env.users]

    plt.figure(figsize=(6, 6))
    plt.scatter(users[:, 0], users[:, 1], c='blue', marker='o', label='Users')
    plt.plot(traj[:, 0], traj[:, 1], c='red', marker='x', label='UAV path')
    plt.scatter([traj[0, 0]], [traj[0, 1]], c='green', s=100, label='Start')
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], c='black', s=100, label='End')
    plt.scatter([env.destination[0]], [env.destination[1]], c='orange', s=150, marker='*', label='Destination')
    for i, center in enumerate(cluster_centers):
        circle = plt.Circle((center[0], center[1]), CLUSTER_DRAW_RADIUS, color='purple', fill=False, linestyle='--', linewidth=2, label='Cluster' if i==0 else None)
        plt.gca().add_patch(circle)
    plt.title(f"UAV Trajectory - {title}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_agent(agent_class, episodes=EPISODES):
    env = UAVEnv()
    agent = agent_class(state_dim=2, action_dim=len(ACTIONS))
    rewards_history = []
    best_reward = -float('inf')
    best_trajectory = None
    best_users = None
    best_destination = None
    epsilon = EPSILON
    min_epsilon = 0.05
    decay = 0.995

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        agent.epsilon = epsilon 
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, int(done))
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
            best_users = env.get_users().copy()
            best_destination = env.destination.copy()
        
        if epsilon > min_epsilon:
            epsilon *= decay
            epsilon = max(epsilon, min_epsilon)

    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{agent_class.__name__} Reward")
    plt.show()

    def plot_best_trajectory(traj, users, destination, title):
        clusters = best_users.reshape(NUM_CLUSTERS, USERS_PER_CLUSTER, 2)
        cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        plt.figure(figsize=(6, 6))
        plt.scatter(users[:, 0], users[:, 1], c='blue', marker='o', label='Users')
        plt.plot(traj[:, 0], traj[:, 1], c='red', marker='x', label='UAV path')
        plt.scatter([traj[0, 0]], [traj[0, 1]], c='green', s=100, label='Start')
        plt.scatter([traj[-1, 0]], [traj[-1, 1]], c='black', s=100, label='End')
        plt.scatter([destination[0]], [destination[1]], c='orange', s=150, marker='*', label='Destination')
        for i, center in enumerate(cluster_centers):
            circle = plt.Circle((center[0], center[1]), CLUSTER_DRAW_RADIUS, color='purple', fill=False, linestyle='--', linewidth=2, label='Cluster' if i==0 else None)
            plt.gca().add_patch(circle)
        plt.title(f"UAV Trajectory (Best Reward) - {title}\nMax Reward: {best_reward:.2f}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

    if best_trajectory is not None:
        plot_best_trajectory(best_trajectory, best_users, best_destination, agent_class.__name__)
    else:
        print("No valid trajectory found.")

if __name__ == "__main__":
    print("Training DQN...")
    train_agent(DQNAgent)

