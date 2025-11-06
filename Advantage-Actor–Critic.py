import numpy as np

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) — From Scratch (NumPy Only).
    Uses:
      - Actor Network: π(a|s) (softmax policy)
      - Critic Network: V(s) (state value estimate)
    Works for discrete action spaces + continuous state inputs.
    """

    def __init__(self, state_size, action_size,
                 actor_hidden=[64, 64], critic_hidden=[64, 64],
                 lr_actor=0.001, lr_critic=0.01, gamma=0.99):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Initialize actor network
        self.aW, self.aB = self._init_net([state_size] + actor_hidden + [action_size])

        # Initialize critic network
        self.cW, self.cB = self._init_net([state_size] + critic_hidden + [1])

    def _init_net(self, layers):
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
             for i in range(len(layers)-1)]
        B = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        return W, B

    def _tanh(self, x): return np.tanh(x)
    def _tanh_d(self, x): return 1 - np.tanh(x)**2

    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x, W, B, softmax=False):
        activations = [x]
        zs = []
        for Wl, Bl in zip(W[:-1], B[:-1]):
            z = activations[-1] @ Wl + Bl
            zs.append(z)
            activations.append(self._tanh(z))
        z = activations[-1] @ W[-1] + B[-1]
        zs.append(z)
        if softmax:
            activations.append(self._softmax(z))
        else:
            activations.append(z)
        return activations, zs

    def act(self, state):
        state = state.reshape(1, -1)
        probs, _ = self.forward(state, self.aW, self.aB, softmax=True)
        return np.random.choice(self.action_size, p=probs[-1].ravel())

    def update(self, state, action, reward, next_state, done):
        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)

        # Critic estimates
        V_s, _ = self.forward(state, self.cW, self.cB)
        V_s = V_s[-1][0, 0]
        V_s2, _ = self.forward(next_state, self.cW, self.cB)
        V_s2 = 0 if done else V_s2[-1][0, 0]

        # Compute advantage
        td_target = reward + self.gamma * V_s2
        advantage = td_target - V_s  # A(s,a)

        # --- Update Critic (MSE) ---
        acts, zs = self.forward(state, self.cW, self.cB)
        grad = np.array([[advantage]])
        grads_W, grads_B = [], []
        for l in reversed(range(len(self.cW))):
            grads_W.insert(0, acts[l].T @ grad)
            grads_B.insert(0, np.sum(grad, axis=0, keepdims=True))
            if l > 0:
                grad = (grad @ self.cW[l].T) * self._tanh_d(zs[l-1])

        for i in range(len(self.cW)):
            self.cW[i] += self.lr_critic * grads_W[i]
            self.cB[i] += self.lr_critic * grads_B[i]

        # --- Update Actor (Policy Gradient using Advantage) ---
        acts, zs = self.forward(state, self.aW, self.aB, softmax=True)
        probs = acts[-1]
        dlog = probs.copy()
        dlog[0, action] -= 1
        grad = advantage * dlog

        grads_W, grads_B = [], []
        for l in reversed(range(len(self.aW))):
            grads_W.insert(0, acts[l].T @ grad)
            grads_B.insert(0, np.sum(grad, axis=0, keepdims=True))
            if l > 0:
                grad = (grad @ self.aW[l].T) * self._tanh_d(zs[l-1])

        for i in range(len(self.aW)):
            self.aW[i] += self.lr_actor * grads_W[i]
            self.aB[i] += self.lr_actor * grads_B[i]

    def train(self, env, episodes=500, max_steps=1000, render=False):
        for ep in range(1, episodes+1):
            state = env.reset()
            total_reward = 0
            for t in range(max_steps):
                if render: env.render()
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            print(f"Episode {ep}/{episodes} | Return: {total_reward:.2f}")
