import numpy as np
import random

class SumTree:
    """
    A binary tree data structure where each leaf node contains a priority value.
    The parent nodes store the sum of their children. Enables efficient
    sampling by priority in O(log N).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using SumTree for sampling.
    """
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # small constant to avoid zero priority

    def add(self, td_error, experience):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_prob = np.array(priorities) / self.tree.total
        is_weights = (1 / (self.tree.size * sampling_prob)) ** beta
        is_weights /= is_weights.max()
        return idxs, batch, is_weights

    def update(self, idx, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.update(idx, priority)


class PrioritizedDQN:
    """
    DQN from scratch with Prioritized Experience Replay (PER).
    """

    def __init__(self, state_size, action_size, hidden_sizes=[64, 64],
                 gamma=0.99, lr=0.001, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, buffer_size=50000, batch_size=64,
                 alpha=0.6, beta=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.beta = beta

        # Initialize neural network
        layer_sizes = [state_size] + hidden_sizes + [action_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)

    def forward(self, x):
        activations, zs = [x], []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ W + b
            zs.append(z)
            activations.append(self.relu(z))
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        activations.append(z)
        return activations, zs

    def backward(self, activations, zs, targets, actions, is_weights):
        q_values = activations[-1]
        batch_indices = np.arange(len(actions))
        td_error = q_values[batch_indices, actions] - targets
        loss_grad = np.zeros_like(q_values)
        loss_grad[batch_indices, actions] = is_weights * td_error

        delta = loss_grad
        grads_W, grads_b = [], []
        for l in reversed(range(len(self.weights))):
            grads_W.insert(0, activations[l].T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if l > 0:
                delta = (delta @ self.weights[l].T) * self.relu_derivative(zs[l - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

        return td_error

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        activations, _ = self.forward(state)
        return np.argmax(activations[-1])

    def remember(self, s, a, r, s2, done):
        # Initial priority is large to ensure early sampling
        self.buffer.add(td_error=1.0, experience=(s, a, r, s2, done))

    def replay(self):
        if self.buffer.tree.size < self.batch_size:
            return
        idxs, minibatch, is_weights = self.buffer.sample(self.batch_size, self.beta)
        states = np.vstack([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.vstack([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        next_acts, _ = self.forward(next_states)
        max_next_q = np.max(next_acts[-1], axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        activations, zs = self.forward(states)
        td_errors = self.backward(activations, zs, targets, actions, is_weights)

        # Update priorities
        for idx, err in zip(idxs, td_errors):
            self.buffer.update(idx, err)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=500, max_steps=500, render=False):
        for e in range(episodes):
            s = env.reset().reshape(1, -1)
            total = 0
            for t in range(max_steps):
                if render: env.render()
                a = self.act(s)
                s2, r, done, _ = env.step(a)
                s2 = s2.reshape(1, -1)
                self.remember(s, a, r, s2, done)
                self.replay()
                s = s2
                total += r
                if done:
                    break
            print(f"Episode {e+1}/{episodes} - Reward: {total:.2f} - Epsilon: {self.epsilon:.3f}")
