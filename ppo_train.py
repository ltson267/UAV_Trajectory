import numpy as np
import matplotlib.pyplot as plt
from config import *
from uav_intercept_env import UAVInterceptEnv
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
    env = UAVInterceptEnv()
    state_dim = 8             
    action_dim = 5            

    agent = PPOIntercept(
        state_dim=state_dim,
        action_dim=action_dim,
        clip_ratio=0.1,
        pi_lr=5e-5,
        vf_lr=1e-3,
        train_pi_iters=20,
        train_v_iters=20,
        lam=0.95,
        entropy_coef=0.08
    )

    HORIZON = 1024
    MAX_EPISODES = 1200

    ep_rewards = []

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

            buf.store(state, a, r, done, logp, v)
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
        print(f"[PPO] Episode {ep+1:03d} | steps={steps:3d} | reward={total_r:9.3f}")

    import matplotlib.pyplot as plt
    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward (UAV Interceptor)")
    plt.show()

if __name__ == "__main__":
    main()
