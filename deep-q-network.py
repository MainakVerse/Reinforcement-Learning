import numpy as np
import random
from collections import deque

class DQNAgent:
    """
    Deep Q-Network (DQN) from scratch (no deep learning libraries)
    Uses a small fully-connected neural net with manual forward/backward propagation.
    """

    def __init__(self, state_size, action_size, hidden_sizes=[64, 64],
                 gamma=0.99, lr=0.001, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=50000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Initialize neural network weights
        layer_sizes = [state_size] + hidden_sizes + [action_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """Forward pass through the network."""
        activations, zs = [x], []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], W) + b
            zs.append(z)
            activations.append(self.relu(z))
        # Output layer (linear)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        activations.append(z)
        return activations, zs

    def backward(self, activations, zs, targets, actions):
        """Backward pass and gradient update."""
        # Compute loss gradient wrt Q-values
        q_values = activations[-1]
        target_q = q_values.copy()
        batch_indices = np.arange(len(actions))
        target_q[batch_indices, actions] = targets
        loss_grad = (q_values - target_q) / len(actions)

        # Backpropagation
        delta = loss_grad
        grads_W, grads_b = [], []
        for l in reversed(range(len(self.weights))):
            grads_W.insert(0, np.dot(activations[l].T, delta))
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(zs[l - 1])

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values, _ = self.forward(state)
        return np.argmax(q_values[-1])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the DQN using random samples from memory."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.vstack([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.vstack([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Compute target Q-values
        next_q_values, _ = self.forward(next_states)
        targets = rewards + self.gamma * np.max(next_q_values[-1], axis=1) * (1 - dones)

        # Backpropagation
        activations, zs = self.forward(states)
        self.backward(activations, zs, targets, actions)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=500, max_steps=500, render=False):
        """Main training loop."""
        for e in range(episodes):
            state = env.reset().reshape(1, -1)
            total_reward = 0
            for step in range(max_steps):
                if render:
                    env.render()
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape(1, -1)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
                if done:
                    break
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
