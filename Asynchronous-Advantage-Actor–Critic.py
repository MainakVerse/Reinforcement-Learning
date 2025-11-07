import numpy as np
import threading
import time

class A3CGlobal:
    """Stores & updates shared Actor and Critic weights across workers."""
    
    def __init__(self, state_size, action_size, hidden=[64, 64], lr_actor=0.0007, lr_critic=0.01, gamma=0.99):
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.action_size = action_size
        self.state_size = state_size

        # Actor Network Parameters (shared)
        self.aW, self.aB = self._init_net([state_size] + hidden + [action_size])
        # Critic Network Parameters (shared)
        self.cW, self.cB = self._init_net([state_size] + hidden + [1])

        # Thread Locks for safe updates
        self.lock = threading.Lock()

    def _init_net(self, layers):
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]) for i in range(len(layers)-1)]
        B = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        return W, B

    def _softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward_actor(self, s):
        a = s
        for i in range(len(self.aW)-1):
            a = np.tanh(a @ self.aW[i] + self.aB[i])
        logits = a @ self.aW[-1] + self.aB[-1]
        return self._softmax(logits[0])

    def forward_critic(self, s):
        a = s
        for i in range(len(self.cW)-1):
            a = np.tanh(a @ self.cW[i] + self.cB[i])
        return (a @ self.cW[-1] + self.cB[-1])[0,0]

    def choose_action(self, state):
        probs = self.forward_actor(state.reshape(1,-1))
        return np.random.choice(self.action_size, p=probs)

    # === Shared gradient update ===
    def update(self, states, actions, returns):
        """Update both actor and critic using computed returns (TD-targets)."""
        with self.lock:
            for s, a, G in zip(states, actions, returns):
                s = s.reshape(1,-1)

                # Critic TD error
                V_s = self.forward_critic(s)
                td_error = G - V_s

                # ---- Critic Update ----
                aC = s
                zsC = []
                actsC = [s]
                for W,b in zip(self.cW[:-1], self.cB[:-1]):
                    z = aC @ W + b
                    zsC.append(z)
                    aC = np.tanh(z)
                    actsC.append(aC)
                z = aC @ self.cW[-1] + self.cB[-1]
                zsC.append(z)
                actsC.append(z)

                grad = np.array([[td_error]])
                for l in reversed(range(len(self.cW))):
                    gW = actsC[l].T @ grad
                    gB = grad
                    self.cW[l] += self.lr_critic * gW
                    self.cB[l] += self.lr_critic * gB
                    if l > 0:
                        grad = (grad @ self.cW[l].T) * (1 - np.tanh(zsC[l-1])**2)

                # ---- Actor Update (Policy Gradient) ----
                probs = self.forward_actor(s)
                dlog = probs.copy()
                dlog[a] -= 1
                grad_policy = td_error * dlog.reshape(1,-1)

                aA = s
                zsA = []
                actsA = [s]
                for W,b in zip(self.aW[:-1], self.aB[:-1]):
                    z = aA @ W + b
                    zsA.append(z)
                    aA = np.tanh(z)
                    actsA.append(aA)
                zsA.append(aA @ self.aW[-1] + self.aB[-1])
                actsA.append(self._softmax(zsA[-1]))

                grad = grad_policy
                for l in reversed(range(len(self.aW))):
                    gW = actsA[l].T @ grad
                    gB = grad
                    self.aW[l] += self.lr_actor * gW
                    self.aB[l] += self.lr_actor * gB
                    if l > 0:
                        grad = (grad @ self.aW[l].T) * (1 - np.tanh(zsA[l-1])**2)



class A3CWorker(threading.Thread):
    """Worker thread — interacts with environment and updates global model."""

    def __init__(self, global_net, env_fn, max_steps=2000):
        threading.Thread.__init__(self)
        self.global_net = global_net
        self.env = env_fn()
        self.max_steps = max_steps

    def run(self):
        for episode in range(1000000):   # Keeps running until terminated externally
            s = self.env.reset()
            states, actions, rewards = [], [], []

            for t in range(self.max_steps):
                a = self.global_net.choose_action(s)
                s2, r, done, _ = self.env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s2
                if done:
                    break

            # Compute discounted returns
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.global_net.gamma * G
                returns.append(G)
            returns.reverse()
            returns = np.array(returns)

            self.global_net.update(states, actions, returns)



def train_a3c(env_fn, state_size, action_size, workers=4):
    global_agent = A3CGlobal(state_size, action_size)
    threads = [A3CWorker(global_agent, env_fn) for _ in range(workers)]
    for t in threads:
        t.start()
    return global_agent
