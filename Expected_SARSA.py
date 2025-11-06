def expected_sarsa(env,
                   alpha: float = 0.1,
                   gamma: float = 0.99,
                   epsilon: float = 0.1,
                   episodes: int = 1000,
                   max_steps: int | None = None,
                   epsilon_decay: float | None = None,
                   min_epsilon: float = 0.01,
                   seed: int | None = None,
                   render: bool = False,
                   verbose: bool = False):
    """
    Tabular Expected SARSA (on-policy) for discrete-action Gym-like envs.
    Update uses the expected next-state value under the epsilon-greedy policy:
      Q(s,a) <- Q(s,a) + α [ r + γ * E_{a'~π}[Q(s',a')] - Q(s,a) ]
    Returns: defaultdict(state -> np.array(action-values))
    """
    import numpy as np
    from collections import defaultdict

    if seed is not None:
        np.random.seed(seed)
        try:
            env.seed(seed)
        except Exception:
            pass

    assert hasattr(env.action_space, "n"), "Action space must be discrete (have .n)"
    nA = env.action_space.n

    def _key(s):
        try:
            if isinstance(s, (int, str)):
                return s
            if hasattr(s, "dtype") and np.isscalar(s):
                return int(s)
            if hasattr(s, "__iter__"):
                return tuple(np.array(s).ravel().tolist())
        except Exception:
            pass
        return s

    Q = defaultdict(lambda: np.zeros(nA, dtype=float))

    def eps_greedy_probs(q_row, eps):
        """Action probabilities under epsilon-greedy for given Q(s,·)."""
        probs = np.full(nA, eps / nA, dtype=float)
        best = np.argmax(q_row)
        probs[best] += 1.0 - eps
        return probs

    for ep in range(1, episodes + 1):
        s = _key(env.reset())
        done = False
        steps = 0
        while not done:
            # choose action (behavior policy: epsilon-greedy from Q)
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))

            s2_raw, r, done, info = env.step(a)
            s2 = _key(s2_raw)

            # Expected value under epsilon-greedy at s'
            if not done:
                probs = eps_greedy_probs(Q[s2], epsilon)
                exp_next = float((probs * Q[s2]).sum())
            else:
                exp_next = 0.0

            td_target = r + gamma * exp_next
            td_error = td_target - Q[s][a]
            Q[s][a] += alpha * td_error

            s = s2
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        if epsilon_decay is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep % max(1, episodes // 10) == 0):
            print(f"Episode {ep}/{episodes} - epsilon: {epsilon:.4f}")

    return Q
