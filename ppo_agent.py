import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from config import *
import math

def mlp(x, sizes, activation=tf.nn.relu, last_activation=None):
    for i, size in enumerate(sizes):
        x = tf.layers.dense(x, size, activation=activation if i < len(sizes)-1 else last_activation)
    return x

class PPOIntercept:
    def __init__(self, state_dim, action_dim,
                 pi_sizes=(256,256), vf_sizes=(256,256),
                 clip_ratio=0.2, pi_lr=1e-3, vf_lr=1e-3,
                 train_pi_iters=80, train_v_iters=40,
                 lam=0.95, max_grad_norm=0.5, entropy_coef=0.18, seed=SEED):
        self.state_dim = state_dim
        self.action_dim = action_dim
        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Placeholders
        self.obs_ph   = tf.placeholder(tf.float32, [None, state_dim], name="obs")
        self.act_ph   = tf.placeholder(tf.int32,   [None], name="act")
        self.adv_ph   = tf.placeholder(tf.float32, [None], name="adv")
        self.ret_ph   = tf.placeholder(tf.float32, [None], name="ret")
        self.logp_old = tf.placeholder(tf.float32, [None], name="logp_old")

        # Policy (categorical) & Value function
        with tf.variable_scope("pi"):
            logits = mlp(self.obs_ph, list(pi_sizes)+[action_dim], activation=tf.nn.relu, last_activation=None)
            self.pi = tf.nn.softmax(logits)

            # logarit prob (log xác xuất của tất cả action tính từ output mạng policy - logits)
            # mỗi phần tử là log(pi(a|s)) cho từng action a
            logp_all = tf.nn.log_softmax(logits)

            # logarit prob của action đã chọn
            self.logp = tf.reduce_sum(tf.one_hot(self.act_ph, action_dim) * logp_all, axis=1)
            # entropy (khuyến khích đa dạng)
            self.entropy = -tf.reduce_mean(tf.reduce_sum(self.pi * logp_all, axis=1))
        with tf.variable_scope("vf"):
            self.v = tf.squeeze(mlp(self.obs_ph, list(vf_sizes)+[1], activation=tf.nn.relu, last_activation=None), axis=1)

        # PPO Clip objective
        ratio = tf.exp(self.logp - self.logp_old)
        clip_adv = tf.clip_by_value(ratio, 1.0-clip_ratio, 1.0+clip_ratio) * self.adv_ph
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, clip_adv)) - entropy_coef * self.entropy
        
        # Value loss
        self.v_loss = tf.reduce_mean(tf.square(self.ret_ph - self.v))
        
        # Optimizers (clip gradien)
        pi_opt = tf.train.AdamOptimizer(pi_lr)
        vf_opt = tf.train.AdamOptimizer(vf_lr)

        pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pi")
        vf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vf")

        self.pi_grads_and_vars = pi_opt.compute_gradients(self.pi_loss, var_list=pi_vars)
        self.vf_grads_and_vars = vf_opt.compute_gradients(self.v_loss, var_list=vf_vars)

        if max_grad_norm is not None:
            self.pi_grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) if g is not None else (g, v)
                                      for (g, v) in self.pi_grads_and_vars]
            self.vf_grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) if g is not None else (g, v)
                                      for (g, v) in self.vf_grads_and_vars]

        self.train_pi = pi_opt.apply_gradients(self.pi_grads_and_vars)
        self.train_v  = vf_opt.apply_gradients(self.vf_grads_and_vars)

        self.train_pi_iters = train_pi_iters
        self.train_v_iters  = train_v_iters
        self.lam = lam

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        pi, v = self.sess.run([self.pi, self.v], {self.obs_ph: [state]})
        probs = pi[0]
        if np.random.rand() < 0.5:
            a = np.argmax(probs)  # greedy
        else:
            a = np.random.choice(self.action_dim, p=probs)  # sample
        logp = math.log(max(probs[a], 1e-8))
        return int(a), float(logp), float(v[0])

    def value(self, state):
        v = self.sess.run(self.v, {self.obs_ph: [state]})
        return float(v[0])

    def update(self, obs, act, adv, ret, logp_old):
        # chuẩn hóa advantage
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-5)

        feed = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.adv_ph: adv,
            self.ret_ph: ret,
            self.logp_old: logp_old
        }

        # train policy nhiều lượt
        for _ in range(self.train_pi_iters):
            self.sess.run(self.train_pi, feed)

        # train value nhiều lượt
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed)

        # Debug
        pi_probs = self.sess.run(self.pi, {self.obs_ph: obs})
        action_mean = pi_probs.mean(axis=0)
        entropy_val = self.sess.run(self.entropy, {self.obs_ph: obs})
        print("[DEBUG] Advantage mean:", adv.mean(), "std:", adv.std())
        print("[DEBUG] Action probs mean:", action_mean)
        print("[DEBUG] Entropy:", entropy_val)
        print("[DEBUG] Reward batch mean:", np.mean(ret), "min:", np.min(ret), "max:", np.max(ret))

class RolloutBuffer:
    def __init__(self, gamma, lam, state_dim):
        self.gamma = gamma
        self.lam = lam
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.obs  = []
        self.act  = []
        self.rew  = []
        self.done = []
        self.logp = []
        self.val  = []

    def store(self, o, a, r, d, logp, v):
        self.obs.append(o)
        self.act.append(a)
        self.rew.append(r)
        self.done.append(d)
        self.logp.append(logp)
        self.val.append(v)

    #Tính advantage theo GAE-λ
    def compute_adv_ret(self, last_val=0.0, last_done=True):
        T = len(self.rew)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)

        next_val = last_val
        next_adv = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(self.done[t])
            delta = self.rew[t] + self.gamma * next_val * nonterminal - self.val[t]
            adv[t] = delta + self.gamma * self.lam * nonterminal * next_adv
            next_adv = adv[t]
            next_val = self.val[t]
        ret = np.array(self.val, dtype=np.float32) + adv
        return (np.array(self.obs, dtype=np.float32),
                np.array(self.act, dtype=np.int32),
                adv.astype(np.float32),
                ret.astype(np.float32),
                np.array(self.logp, dtype=np.float32))
