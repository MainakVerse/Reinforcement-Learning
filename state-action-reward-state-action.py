def sarsa(env,
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
    Tabular SARSA (on-policy) for discrete-action OpenAI Gym-like environments.
    Returns: Q (defaultdict mapping state -> np.array of action-values)

    Parameters are analogous to typical tabular RL functions:
    - env: environment with reset(), step(action) and action_space.n
    - alpha: learning rate
    - gamma: discount factor
    - epsilon: initial epsilon for epsilon-greedy policy
    - episodes: number of episodes to train
    - max_steps: optional max steps per episode
    - epsilon_decay: multiply epsilon by this each episode (if provided)
    - min_epsilon: lower bound when decaying
    - seed: RNG seed (tries env.seed if available)
    - render: call env.render() each step if True
    - verbose: print progress every 10% of episodes if True
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

    def _state_key(s):
        """Convert state to a hashable key (int, tuple, etc.)."""
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

    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    for ep in range(1, episodes + 1):
        obs = env.reset()
        state = _state_key(obs)

        # choose initial action using epsilon-greedy (on-policy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        done = False
        steps = 0
        while not done:
            if render:
                try:
                    env.render()
                except Exception:
                    pass

            next_obs, reward, done, info = env.step(action)
            next_state = _state_key(next_obs)

            # choose next action using current policy (epsilon-greedy)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.argmax(Q[next_state]))

            # SARSA update (on-policy TD):
            # Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
            td_target = reward + (gamma * Q[next_state][next_action] if not done else 0.0)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            action = next_action
            steps += 1
            if (max_steps is not None) and (steps >= max_steps):
                break

        # decay epsilon if provided
        if epsilon_decay is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep % max(1, episodes // 10) == 0):
            print(f"Episode {ep}/{episodes} - epsilon: {epsilon:.4f}")

    return Q
