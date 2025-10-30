import numpy as np
import random
from collections import deque

class DoubleDQNAgent:
    """
    Double Deep Q-Network (Double DQN) from scratch using NumPy only.
    Uses two neural networks (online + target) for stable value estimation.
    """

    def __init__(self, state_size, action_size, hidden_sizes=[64, 64],
                 gamma=0.99, lr=0.001, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=50000, batch_size=64,
                 target_update_freq=50):

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

        # Online and target networks
        self.online_weights, self.online_biases = self._init_network(hidden_sizes)
        self.target_weights, self.target_biases = self._copy_network()

        self.steps = 0

    def _init_network(self, hidden_sizes):
        """Initialize a feedforward neural network."""
        layer_sizes = [self.state_size] + hidden_sizes + [self.action_size]
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
                   for i in range(len(layer_sizes) - 1)]
        biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        return weights, biases

    def _copy_network(self):
        """Deep copy the online network to initialize the target network."""
        weights = [w.copy() for w in self.online_weights]
        biases = [b.copy() for b in self.online_biases]
        return weights, biases

    def _update_target_network(self):
        """Synchronize target network with online network."""
        self.target_weights = [w.copy() for w in self.online_weights]
        self.target_biases = [b.copy() for b in self.online_biases]

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)

    def _forward(self, x, weights, biases):
        """Forward pass through network."""
        activations, zs = [x], []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = np.dot(activations[-1], W) + b
            zs.append(z)
            activations.append(self.relu(z))
        z = np.dot(activations[-1], weights[-1]) + biases[-1]
        zs.append(z)
        activations.append(z)
        return activations, zs

    def _backward(self, activations, zs, targets, actions):
        """Backward pass for Q-value updates."""
        q_values = activations[-1]
        batch_indices = np.arange(len(actions))
        target_q = q_values.copy()
        target_q[batch_indices, actions] = targets
        loss_grad = (q_values - target_q) / len(actions)

        delta = loss_grad
        grads_W, grads_b = [], []
        for l in reversed(range(len(self.online_weights))):
            grads_W.insert(0, np.dot(activations[l].T, delta))
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if l > 0:
                delta = np.dot(delta, self.online_weights[l].T) * self.relu_derivative(zs[l - 1])

        for i in range(len(self.online_weights)):
            self.online_weights[i] -= self.lr * grads_W[i]
            self.online_biases[i] -= self.lr * grads_b[i]

    def act(self, state):
        """Epsilon-greedy action."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        activations, _ = self._forward(state, self.online_weights, self.online_biases)
        return np.argmax(activations[-1])

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def replay(self):
        """Sample batch and update the online network."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.vstack([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Online net selects best actions for next state
        online_next, _ = self._forward(next_states, self.online_weights, self.online_biases)
        next_actions = np.argmax(online_next[-1], axis=1)

        # Target net estimates the Q-value for those actions
        target_next, _ = self._forward(next_states, self.target_weights, self.target_biases)
        target_values = target_next[-1][np.arange(self.batch_size), next_actions]

        # Compute target Q-values
        targets = rewards + self.gamma * target_values * (1 - dones)

        # Update online network using gradient descent
        activations, zs = self._forward(states, self.online_weights, self.online_biases)
        self._backward(activations, zs, targets, actions)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network every few steps
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self._update_target_network()

    def train(self, env, episodes=500, max_steps=500, render=False):
        """Main Double DQN training loop."""
        for e in range(episodes):
            state = env.reset().reshape(1, -1)
            total_reward = 0
            for step in range(max_steps):
                if render: env.render()
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape(1, -1)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
                if done: break
            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
