import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def discount_rewards(rewards, gamma):
    G = 0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), dtype=float)


class TRPO:
    """
    TRPO (Trust Region Policy Optimization) — NumPy Only
    Discrete Action Policy + Value Baseline
    """

    def __init__(self, state_size, action_size,
                 hidden=[64, 64], gamma=0.99,
                 delta=0.01, cg_iters=10, backtrack=0.8, backtrack_iters=20):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.delta = delta
        self.cg_iters = cg_iters
        self.backtrack = backtrack
        self.backtrack_iters = backtrack_iters

        layer_sizes = [state_size] + hidden + [action_size]
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
                  for i in range(len(layer_sizes)-1)]
        self.B = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

        # Value function parameters
        self.vw = np.random.randn(state_size, 1) * 0.1
        self.vb = np.zeros((1,1))

    def forward(self, X):
        A = X
        for W, B in zip(self.W[:-1], self.B[:-1]):
            A = np.tanh(A @ W + B)
        logits = A @ self.W[-1] + self.B[-1]
        return softmax(logits)

    def value(self, X):
        return (X @ self.vw + self.vb).ravel()

    def update_value(self, states, returns, lr=0.01, iters=50):
        for _ in range(iters):
            v = self.value(states)
            grad_w = states.T @ (v - returns)[:,None]
            grad_b = np.sum((v - returns))
            self.vw -= lr * grad_w
            self.vb -= lr * grad_b

    def policy_gradient(self, states, actions, advantages):
        probs = self.forward(states)
        logp = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        grad_coeff = advantages[:,None]

        grads_W = [np.zeros_like(W) for W in self.W]
        grads_B = [np.zeros_like(B) for B in self.B]

        for i in range(len(states)):
            s = states[i:i+1]
            p = probs[i:i+1]
            y = np.zeros_like(p)
            y[0, actions[i]] = 1
            dl = (p - y) * advantages[i]

            A = [s]
            for W,B in zip(self.W[:-1], self.B[:-1]):
                A.append(np.tanh(A[-1] @ W + B))

            grads_W[-1] += A[-1].T @ dl
            grads_B[-1] += dl

            delta = dl @ self.W[-1].T * (1 - A[-2]**2)
            for l in reversed(range(len(self.W)-1)):
                grads_W[l] += A[l].T @ delta
                grads_B[l] += delta
                if l > 0:
                    delta = (delta @ self.W[l].T) * (1 - A[l-1]**2)

        return [g/len(states) for g in grads_W], [g/len(states) for g in grads_B]

    def flat(self, W, B):
        return np.concatenate([W.flatten() for W in W] + [B.flatten() for B in B])

    def unflat(self, vec):
        Wnew, Bnew = [], []
        idx = 0
        for W,B in zip(self.W, self.B):
            wsize = W.size
            bsize = B.size
            Wnew.append(vec[idx:idx+wsize].reshape(W.shape))
            idx += wsize
            Bnew.append(vec[idx:idx+bsize].reshape(B.shape))
            idx += bsize
        return Wnew, Bnew

    def fisher_vector_product(self, states, v):
        eps = 1e-5
        Wp, Bp = self.unflat(v)
        old_probs = self.forward(states)

        self.W = [W+eps*Wp[i] for i,W in enumerate(self.W)]
        self.B = [B+eps*Bp[i] for i,B in enumerate(self.B)]
        new_probs = self.forward(states)
        self.W = [W-eps*Wp[i] for i,W in enumerate(self.W)]
        self.B = [B-eps*Bp[i] for i,B in enumerate(self.B)]

        kl = np.sum(old_probs * (np.log(old_probs+1e-8) - np.log(new_probs+1e-8)), axis=1)
        grad = np.mean(kl)
        return v * grad

    def conjugate_gradient(self, states, g):
        x = np.zeros_like(g)
        r = g.copy()
        p = r.copy()
        rr = r @ r

        for _ in range(self.cg_iters):
            Ap = self.fisher_vector_product(states, p)
            alpha = rr / (p @ Ap + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rr_new = r @ r
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    def line_search(self, states, actions, advantages, step, fullstep):
        old_W, old_B = [W.copy() for W in self.W], [B.copy() for B in self.B]
        for _ in range(self.backtrack_iters):
            scale = self.backtrack
            self.W = [W + scale*F for W,F in zip(old_W, fullstep[0])]
            self.B = [B + scale*F for B,F in zip(old_B, fullstep[1])]
            new_probs = self.forward(states)
            old_probs = softmax(states @ old_W[-1] + old_B[-1])
            kl = np.mean(np.sum(old_probs * (np.log(old_probs+1e-8) - np.log(new_probs+1e-8)), axis=1))
            if kl <= self.delta:
                return
        self.W, self.B = old_W, old_B  # revert

    def train(self, env, episodes=200):
        for ep in range(1, episodes+1):
            s = env.reset()
            states, actions, rewards = [], [], []
            done = False
            while not done:
                probs = self.forward(s.reshape(1,-1))[0]
                a = np.random.choice(self.action_size, p=probs)
                s2, r, done, _ = env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s2

            states = np.array(states)
            returns = discount_rewards(rewards, self.gamma)
            values = self.value(states)
            advantages = returns - values

            # Update Critic
            self.update_value(states, returns)

            # Policy Gradient
            gW, gB = self.policy_gradient(states, actions, advantages)
            g = self.flat(gW, gB)

            # Solve for step direction
            x = self.conjugate_gradient(states, g)
            step = x / (np.sqrt(x @ self.fisher_vector_product(states, x)) / np.sqrt(self.delta))

            # Apply TRPO update via line search
            fullstep = self.unflat(step)
            self.line_search(states, actions, advantages, step, fullstep)

            print(f"Episode {ep}/{episodes} | Return: {sum(rewards):.2f}")
