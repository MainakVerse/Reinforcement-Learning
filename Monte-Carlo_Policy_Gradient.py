import numpy as np

class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) - From Scratch (NumPy Only)
    Works on environments with continuous state (vector) + discrete action space.
    """

    def __init__(self, state_size, action_size, hidden_sizes=[32, 32],
                 lr=0.01, gamma=0.99):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

        layer_sizes = [state_size] + hidden_sizes + [action_size]

        # Fully-connected network parameters
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            for i in range(len(layer_sizes)-1)
        ]
        self.biases = [
            np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)
        ]

    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x):
        activations = [x]
        zs = []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ W + b
            zs.append(z)
            activations.append(np.tanh(z))
        # Output logits → softmax
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        probs = self._softmax(z)
        activations.append(probs)
        return activations, zs

    def act(self, state):
        state = state.reshape(1, -1)
        probs, _ = self.forward(state)
        probs = probs[-1].ravel()
        return np.random.choice(self.action_size, p=probs)

    def compute_returns(self, rewards):
        G = 0
        discounted = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted.append(G)
        discounted.reverse()
        return np.array(discounted, dtype=float)

    def update(self, states, actions, returns):
        # Convert to matrices
        states = np.vstack(states)
        actions = np.array(actions)

        # Forward pass to get action probabilities
        activations, zs = self.forward(states)
        probs = activations[-1]

        # Policy gradient log-prob derivative
        grads = probs.copy()
        grads[np.arange(len(actions)), actions] -= 1
        grads /= len(actions)
        grads *= returns.reshape(-1, 1)

        delta = grads
        grads_W = []
        grads_b = []

        for l in reversed(range(len(self.weights))):
            a_prev = activations[l]
            grads_W.insert(0, a_prev.T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))

            if l > 0:
                delta = (delta @ self.weights[l].T) * (1 - np.tanh(zs[l-1])**2)

        # Gradient descent update
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def train(self, env, episodes=500, max_steps=1000, render=False):
        for ep in range(1, episodes+1):
            state = env.reset()
            states, actions, rewards = [], [], []
            for t in range(max_steps):
                if render: env.render()

                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                if done: break

            returns = self.compute_returns(rewards)
            self.update(states, actions, returns)

            print(f"Episode {ep}/{episodes} | Return: {sum(rewards):.2f}")
