import numpy as np
import random
from collections import deque

# =========================
# Utility: Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones   = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return ( self.states[idx],
                 self.actions[idx],
                 self.rewards[idx],
                 self.next_states[idx],
                 self.dones[idx] )

# =========================
# Utility: OU Noise
# =========================
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.action_dim)
        self.x += dx
        return self.x

# =========================
# Minimal MLP with manual backprop (tanh hidden, linear out)
# =========================
class MLP:
    def __init__(self, layer_sizes, out_activation=None):
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
                  for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.out_activation = out_activation  # None or 'tanh' or 'linear'

    def forward(self, X):
        A = [X]     # activations per layer (including input)
        Z = []      # pre-activations per hidden layer
        for i in range(len(self.W)-1):
            z = A[-1] @ self.W[i] + self.b[i]
            Z.append(z)
            A.append(np.tanh(z))
        # last layer
        zL = A[-1] @ self.W[-1] + self.b[-1]
        Z.append(zL)
        if self.out_activation == 'tanh':
            out = np.tanh(zL)
        else:
            out = zL
        A.append(out)
        return A, Z

    # Backprop for scalar loss gradient at output (dL/dOut supplied)
    def backward(self, A, Z, dOut):
        grads_W = [np.zeros_like(W) for W in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]

        # derivative through output activation
        if self.out_activation == 'tanh':
            dZ = dOut * (1 - np.tanh(Z[-1])**2)
        else:  # linear
            dZ = dOut

        # last layer grads
        grads_W[-1] = A[-2].T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        # hidden layers
        dA_prev = dZ @ self.W[-1].T
        for l in reversed(range(len(self.W)-1)):
            dZ = dA_prev * (1 - np.tanh(Z[l])**2)
            grads_W[l] = A[l].T @ dZ
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True)
            if l > 0:
                dA_prev = dZ @ self.W[l].T

        return grads_W, grads_b

    def apply_grads(self, gW, gb, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * gW[i]
            self.b[i] -= lr * gb[i]

# =========================
# DDPG Agent (NumPy only)
# =========================
class DDPGAgent:
    """
    Deep Deterministic Policy Gradient — NumPy-only implementation.
    - Deterministic Actor μ(s) with tanh output (scaled to action bounds)
    - Critic Q(s,a) (scalar)
    - Target networks with soft update τ
    - Replay buffer + OU exploration noise
    """

    def __init__(self,
                 state_dim, action_dim,
                 action_low, action_high,
                 actor_hidden=[256, 256],
                 critic_hidden=[256, 256],
                 gamma=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3,
                 buffer_size=100000, batch_size=64,
                 ou_theta=0.15, ou_sigma=0.2, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size

        self.act_low = np.array(action_low, dtype=np.float32)
        self.act_high = np.array(action_high, dtype=np.float32)
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_mid   = (self.act_high + self.act_low) / 2.0

        # Actor: s -> a in [-1,1]^dim (tanh), then scaled to bounds
        self.actor = MLP([self.s_dim] + actor_hidden + [self.a_dim], out_activation='tanh')
        self.actor_tgt = MLP([self.s_dim] + actor_hidden + [self.a_dim], out_activation='tanh')
        self._hard_update(self.actor_tgt, self.actor)

        # Critic: (s,a) -> Q scalar (linear out)
        self.critic = MLP([self.s_dim + self.a_dim] + critic_hidden + [1], out_activation=None)
        self.critic_tgt = MLP([self.s_dim + self.a_dim] + critic_hidden + [1], out_activation=None)
        self._hard_update(self.critic_tgt, self.critic)

        # Replay + noise
        self.buffer = ReplayBuffer(buffer_size, self.s_dim, self.a_dim)
        self.noise = OUNoise(self.a_dim, theta=ou_theta, sigma=ou_sigma)

    # -------- utilities --------
    def _hard_update(self, target, source):
        for i in range(len(source.W)):
            target.W[i] = source.W[i].copy()
            target.b[i] = source.b[i].copy()

    def _soft_update(self, target, source):
        for i in range(len(source.W)):
            target.W[i] = (1 - self.tau) * target.W[i] + self.tau * source.W[i]
            target.b[i] = (1 - self.tau) * target.b[i] + self.tau * source.b[i]

    def _scale_to_bounds(self, a_tanh):
        # a in [-1,1] -> scale to [low, high]
        return self.act_mid + self.act_scale * a_tanh

    # -------- interaction --------
    def select_action(self, state, add_noise=True):
        s = state.reshape(1, -1).astype(np.float32)
        A, Z = self.actor.forward(s)
        a = A[-1][0]  # in [-1,1]
        if add_noise:
            a = a + self.noise.sample()
            a = np.clip(a, -1.0, 1.0)
        return self._scale_to_bounds(a)

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, float(done))

    # -------- training step --------
    def train_step(self):
        if self.buffer.size < self.batch_size:
            return None

        S, A, R, S2, D = self.buffer.sample(self.batch_size)  # shapes: (B,sd), (B,ad), (B,1), (B,sd), (B,1)

        # ----- Critic target -----
        # a' = μ_target(s')
        A2_act, _ = self.actor_tgt.forward(S2)
        A2 = A2_act[-1]                                   # in [-1,1]
        A2_scaled = self._scale_to_bounds(A2)

        # Q' = Q_target(s', a')
        SA2 = np.concatenate([S2, A2_scaled], axis=1)
        Q2_act, _ = self.critic_tgt.forward(SA2)
        Q2 = Q2_act[-1]                                   # (B,1)

        y = R + self.gamma * (1 - D) * Q2                 # targets (B,1)

        # ----- Critic update (MSE) -----
        SA = np.concatenate([S, A], axis=1)
        Q_act, Zc = self.critic.forward(SA)
        Q = Q_act[-1]                                     # (B,1)
        err = Q - y                                       # dL/dQ
        dQ = err / len(S)                                 # mean

        gW_c, gB_c = self.critic.backward(Q_act, Zc, dQ)
        self.critic.apply_grads(gW_c, gB_c, self.critic_lr)

        # ----- Actor update (Deterministic Policy Gradient) -----
        # We need ∂Q/∂a at a = μ(s), then chain with ∂μ/∂θ
        A_act, Za = self.actor.forward(S)
        A_mu = A_act[-1]                                  # in [-1,1]
        A_scaled = self._scale_to_bounds(A_mu)
        SA_mu = np.concatenate([S, A_scaled], axis=1)

        # Forward critic to get gradient wrt its input (s,a)
        C_acts, C_z = self.critic.forward(SA_mu)          # (B,1)
        # dL/dQ for actor objective is -1/B (we maximize Q): so dQ_sign = -1/B
        dQ_actor = -np.ones_like(C_acts[-1]) / len(S)

        # Backprop through critic to input (s,a)
        # First get grads w.r.t last layer pre-activation (since linear out, dZ = dQ_actor)
        # Use critic.backward to get parameter grads (unused) and we also need dInput
        # We'll manually compute dInput by replicating backward’s first step:

        # Backward last layer to get grad on penultimate activation:
        dZ = dQ_actor  # linear out
        dA_prev = dZ @ self.critic.W[-1].T               # gradient wrt penultimate activation

        # Propagate through hidden layers down to input (s,a)
        for l in reversed(range(len(self.critic.W)-1)):
            # derivative of tanh at Z
            dZ = dA_prev * (1 - np.tanh(C_z[l])**2)
            if l > 0:
                dA_prev = dZ @ self.critic.W[l].T
            else:
                dInput = dZ @ self.critic.W[l].T          # gradient wrt input (s,a)

        # Split dInput into (ds, da_scaled). We only need da term (policy gradient)
        dInput_sa = dInput  # (B, s_dim + a_dim)
        dQ_da_scaled = dInput_sa[:, self.s_dim:]          # (B, a_dim)

        # Chain rule: a_scaled = mid + scale * tanh_out  => ∂a_scaled/∂tanh_out = scale
        dQ_da_tanh = dQ_da_scaled * self.act_scale        # elementwise (broadcast)

        # Backprop through actor to parameters using dOut = dQ/da_tanh
        gW_a, gB_a = self.actor.backward(A_act, Za, dQ_da_tanh)  # uses internal tanh/linear chain
        self.actor.apply_grads(gW_a, gB_a, self.actor_lr)

        # ----- Soft update targets -----
        self._soft_update(self.actor_tgt, self.actor)
        self._soft_update(self.critic_tgt, self.critic)

        # Diagnostics
        loss_critic = float(0.5 * np.mean((Q - y)**2))
        return {"critic_loss": loss_critic, "Q_mean": float(Q.mean())}

    # -------- training loop helper --------
    def train(self, env, episodes=500, max_steps=1000, start_steps=1000, explore_noise=True):
        """
        Simple training loop for Gym-like continuous envs.
        """
        total_steps = 0
        for ep in range(1, episodes+1):
            s = env.reset()
            self.noise.reset()
            ep_ret = 0.0
            for t in range(max_steps):
                if total_steps < start_steps:
                    a = env.action_space.sample()
                else:
                    a = self.select_action(s, add_noise=explore_noise)
                s2, r, done, _ = env.step(a)

                self.remember(s, a, r, s2, done)
                s = s2
                ep_ret += r
                total_steps += 1

                # gradient step
                info = self.train_step()

                if done:
                    break
            print(f"Episode {ep}/{episodes} | Return: {ep_ret:.2f}")
