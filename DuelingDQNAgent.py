import numpy as np
import random
from collections import deque

class DuelingDQNAgent:
    """
    Dueling DQN (from scratch, NumPy only).
    Network architecture: shared trunk -> value head (1 output) + advantage head (action_size outputs)
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128],
                 gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=50000, batch_size=64,
                 target_update_freq=200):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        self.steps = 0

        # Build online and target networks (shared trunk + two heads)
        self.trunk_w, self.trunk_b = self._init_layers([state_size] + hidden_sizes)
        # value head: last trunk size -> 1
        self.value_w = np.random.randn(hidden_sizes[-1], 1) * np.sqrt(2. / hidden_sizes[-1])
        self.value_b = np.zeros((1, 1))
        # advantage head: last trunk size -> action_size
        self.adv_w = np.random.randn(hidden_sizes[-1], action_size) * np.sqrt(2. / hidden_sizes[-1])
        self.adv_b = np.zeros((1, action_size))

        # target networks (copies)
        self.t_trunk_w = [w.copy() for w in self.trunk_w]
        self.t_trunk_b = [b.copy() for b in self.trunk_b]
        self.t_value_w = self.value_w.copy()
        self.t_value_b = self.value_b.copy()
        self.t_adv_w = self.adv_w.copy()
        self.t_adv_b = self.adv_b.copy()

    def _init_layers(self, sizes):
        weights = []
        biases = []
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2. / sizes[i])
            b = np.zeros((1, sizes[i + 1]))
            weights.append(w)
            biases.append(b)
        return weights, biases

    def relu(self, x): return np.maximum(0, x)
    def relu_deriv(self, x): return (x > 0).astype(float)

    def _forward_trunk(self, x, weights, biases):
        activations = [x]
        zs = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            zs.append(z)
            activations.append(self.relu(z))
        # last trunk layer (no activation here; treat as trunk output)
        z = activations[-1] @ weights[-1] + biases[-1]
        zs.append(z)
        activations.append(self.relu(z))
        return activations, zs

    def _compute_q_from_trunk(self, trunk_activation, v_w, v_b, a_w, a_b):
        V = trunk_activation @ v_w + v_b                # shape (batch, 1)
        A = trunk_activation @ a_w + a_b                # shape (batch, action_size)
        A_mean = np.mean(A, axis=1, keepdims=True)
        Q = V + (A - A_mean)
        return Q, V, A

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        x = state.reshape(1, -1)
        act_trunk, _ = self._forward_trunk(x, self.trunk_w, self.trunk_b)
        q_vals, _, _ = self._compute_q_from_trunk(act_trunk[-1], self.value_w, self.value_b, self.adv_w, self.adv_b)
        return int(np.argmax(q_vals, axis=1)[0])

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def _update_target(self):
        self.t_trunk_w = [w.copy() for w in self.trunk_w]
        self.t_trunk_b = [b.copy() for b in self.trunk_b]
        self.t_value_w = self.value_w.copy()
        self.t_value_b = self.value_b.copy()
        self.t_adv_w = self.adv_w.copy()
        self.t_adv_b = self.adv_b.copy()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch], dtype=int)
        rewards = np.array([t[2] for t in minibatch], dtype=float)
        next_states = np.vstack([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch], dtype=float)

        # Target Q-values using target network (double-style could be added; here standard target)
        next_trunk_act, _ = self._forward_trunk(next_states, self.t_trunk_w, self.t_trunk_b)
        next_q, _, _ = self._compute_q_from_trunk(next_trunk_act[-1], self.t_value_w, self.t_value_b, self.t_adv_w, self.t_adv_b)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)  # shape (batch,)

        # Forward current trunk to get activations for backprop
        trunk_acts, trunk_zs = self._forward_trunk(states, self.trunk_w, self.trunk_b)
        trunk_out = trunk_acts[-1]  # (batch, hidden_last)

        # Current predictions
        q_pred, V_pred, A_pred = self._compute_q_from_trunk(trunk_out, self.value_w, self.value_b, self.adv_w, self.adv_b)

        # Build gradient for Q (MSE loss): grad_Q = (q_pred - target_one_hot)/batch
        batch_size = states.shape[0]
        grad_q = (q_pred.copy())
        grad_q[np.arange(batch_size), actions] -= targets
        grad_q /= batch_size  # shape (batch, action_size)

        # Backprop into advantage and value heads:
        # Q = V + A - mean(A) -> gradients distribute:
        # dV = grad_q summed over actions
        dV = np.sum(grad_q, axis=1, keepdims=True)                      # (batch,1)
        # dA = grad_q - mean(grad_q over actions)
        dA = grad_q - np.mean(grad_q, axis=1, keepdims=True)            # (batch,action_size)

        # Gradients for head weights/biases
        grad_value_w = trunk_out.T @ dV                                 # (hidden_last,1)
        grad_value_b = np.sum(dV, axis=0, keepdims=True)                # (1,1)

        grad_adv_w = trunk_out.T @ dA                                   # (hidden_last,action_size)
        grad_adv_b = np.sum(dA, axis=0, keepdims=True)                  # (1,action_size)

        # Backprop into trunk: propagate gradients from both heads
        # d_trunk = dV @ v_w.T + dA @ a_w.T, then through ReLU derivatives of trunk layers
        d_trunk = dV @ self.value_w.T + dA @ self.adv_w.T               # (batch, hidden_last)

        # Backprop through trunk layers
        grads_W = []
        grads_b = []
        delta = d_trunk
        for l in reversed(range(len(self.trunk_w))):
            a_prev = trunk_acts[l]                                       # activation of previous layer
            gradW = a_prev.T @ delta                                     # (dims_l, dims_{l+1})
            gradb = np.sum(delta, axis=0, keepdims=True)
            grads_W.insert(0, gradW)
            grads_b.insert(0, gradb)
            if l > 0:
                delta = (delta @ self.trunk_w[l].T) * self.relu_deriv(trunk_zs[l-1])

        # Update parameters (SGD step)
        for i in range(len(self.trunk_w)):
            self.trunk_w[i] -= self.lr * grads_W[i]
            self.trunk_b[i] -= self.lr * grads_b[i]
        self.value_w -= self.lr * grad_value_w
        self.value_b -= self.lr * grad_value_b
        self.adv_w -= self.lr * grad_adv_w
        self.adv_b -= self.lr * grad_adv_b

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target sync
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self._update_target()

    def train(self, env, episodes=500, max_steps=500, render=False):
        for ep in range(1, episodes + 1):
            s = env.reset()
            s = s.reshape(1, -1)
            total = 0
            for t in range(max_steps):
                if render:
                    env.render()
                a = self.act(s)
                s2, r, done, _ = env.step(a)
                s2 = s2.reshape(1, -1)
                self.remember(s, a, r, s2, done)
                s = s2
                total += r
                self.replay()
                if done:
                    break
            print(f"Episode {ep}/{episodes} - Reward: {total:.2f} - Epsilon: {self.epsilon:.3f}")
