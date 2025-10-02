import numpy as np
import matplotlib.pyplot as plt
from config import *
from uav_intercept_env_ppo import UAVInterceptEnvPPO
from ppo_agent import PPOIntercept, RolloutBuffer

def plot_scene(env, title="UAV Intercept (PPO)"):
    traj = env.get_trajectory()
    links = env.get_links()

    plt.figure(figsize=(6,6))
    for (su, du) in links:
        plt.plot([su[0], du[0]], [su[1], du[1]], linewidth=1)
    plt.plot(traj[:,0], traj[:,1], marker='x', linewidth=1)
    plt.scatter([traj[0,0]], [traj[0,1]], s=80)
    plt.title(title)
    plt.xlim(0, GRID_SIZE-1); plt.ylim(0, GRID_SIZE-1)
    plt.grid(True)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show()

def main():
    env = UAVInterceptEnvPPO()
    state_dim = 8             
    action_dim = 5            

    agent = PPOIntercept(
        state_dim=state_dim,
        action_dim=action_dim,
        clip_ratio=0.2,
        pi_lr=5e-4,
        vf_lr=1e-3,
        train_pi_iters=40,
        train_v_iters=40,
        lam=0.95,
        entropy_coef=0.08
    )

    HORIZON = 800  # Gần bằng max_steps để buffer đa dạng hơn
    MAX_EPISODES = 3000

    ep_rewards = []
    best_reward = -float('inf')
    best_trajectory = None
    best_links = None

    for ep in range(MAX_EPISODES):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        env.max_steps = 1200  

        buf = RolloutBuffer(gamma=GAMMA, lam=agent.lam, state_dim=state_dim)

        while not done:
            a, logp, v = agent.act(state)
            ns, r, done = env.step(a)

            # Reward normalization
            if not hasattr(main, 'rew_batch'): main.rew_batch = []
            main.rew_batch.append(r)
            if len(main.rew_batch) > 1000: main.rew_batch.pop(0)
            norm_r = r
            if len(main.rew_batch) > 10:
                max_abs = max(1.0, np.max(np.abs(main.rew_batch)))
                norm_r = r / max_abs
            buf.store(state, a, norm_r, done, logp, v)
            total_r += r
            steps += 1
            state = ns

            if len(buf.rew) >= HORIZON:
                last_v = 0 if done else agent.value(state)
                obs, act, adv, ret, logp_old = buf.compute_adv_ret(last_val=last_v, last_done=done)
                agent.update(obs, act, adv, ret, logp_old)
                buf.reset()

        # Update cuối cùng nếu còn buffer
        if len(buf.rew) > 0:
            obs, act, adv, ret, logp_old = buf.compute_adv_ret(last_val=0.0, last_done=True)
            agent.update(obs, act, adv, ret, logp_old)

        ep_rewards.append(total_r)
        if total_r > best_reward:
            best_reward = total_r
            best_trajectory = env.get_trajectory().copy()
            best_links = env.get_links().copy()
        if len(ep_rewards) >= 10:
            avg10 = np.mean(ep_rewards[-10:])
        else:
            avg10 = np.mean(ep_rewards)
        print(f"[PPO] Episode {ep+1:03d} | steps={steps:3d} | reward={total_r:9.3f} | avg10={avg10:9.3f}")

    import matplotlib.pyplot as plt
    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward (UAV Interceptor)")
    plt.show()

    # Vẽ trajectory của episode có reward lớn nhất
    if best_trajectory is not None and best_links is not None:
        plt.figure(figsize=(6,6))
        for (su, du) in best_links:
            plt.plot([su[0], du[0]], [su[1], du[1]], linewidth=1)
        plt.plot(best_trajectory[:,0], best_trajectory[:,1], marker='x', linewidth=1, label='UAV path')
        plt.scatter([best_trajectory[0,0]], [best_trajectory[0,1]], s=80, c='green', label='Start')
        plt.scatter([best_trajectory[-1,0]], [best_trajectory[-1,1]], s=80, c='red', label='End')
        plt.title(f"Best Trajectory (Reward={best_reward:.2f})")
        plt.xlim(0, GRID_SIZE[0]-1)
        plt.ylim(0, GRID_SIZE[1]-1)
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
