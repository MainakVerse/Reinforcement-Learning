def q_learning(env,
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
    Tabular Q-Learning (from scratch) for OpenAI Gym-like environments with discrete action space.
    - Works when env.observation_space is discrete (int) or when state is representable as a hashable tuple.
    - Returns: Q (defaultdict mapping state -> numpy array of action-values)
    
    Parameters
    ----------
    env : gym.Env
        Environment with `reset()` and `step(action)` methods and `action_space.n`.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial epsilon for epsilon-greedy.
    episodes : int
        Number of episodes to train.
    max_steps : int or None
        Max steps per episode (None => until done).
    epsilon_decay : float or None
        Multiply epsilon by this after each episode if provided.
    min_epsilon : float
        Lower bound for epsilon when decaying.
    seed : int or None
        Random seed (sets numpy RNG and env.seed if available).
    render : bool
        Call env.render() each step if True.
    verbose : bool
        Print progress every 10% of episodes if True.
    """
    import numpy as np
    from collections import defaultdict

    if seed is not None:
        np.random.seed(seed)
        try:
            env.seed(seed)
        except Exception:
            pass

    # Ensure discrete action space
    assert hasattr(env.action_space, "n"), "Action space must be discrete (have .n)"

    def _state_key(s):
        """Convert state to a hashable key (int or tuple)."""
        try:
            # common case: discrete integer observation
            if isinstance(s, (int, str)):
                return s
            # numpy scalar
            if hasattr(s, "dtype") and np.isscalar(s):
                return int(s)
            # numpy array or list -> tuple
            if hasattr(s, "__iter__"):
                return tuple(np.array(s).ravel().tolist())
        except Exception:
            pass
        # fallback to raw
        return s

    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    for ep in range(1, episodes + 1):
        obs = env.reset()
        state = _state_key(obs)
        done = False
        steps = 0
        while not done:
            # epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, done, info = env.step(action)
            if render:
                try:
                    env.render()
                except Exception:
                    pass

            next_state = _state_key(next_obs)

            # Q-Learning update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            best_next = np.max(Q[next_state]) if not done else 0.0
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            steps += 1
            if (max_steps is not None) and (steps >= max_steps):
                break

        # decay epsilon if requested
        if epsilon_decay is not None:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep % max(1, episodes // 10) == 0):
            print(f"Episode {ep}/{episodes} - epsilon: {epsilon:.4f}")

    return Q
