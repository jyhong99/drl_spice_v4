"""Utility helpers for training loops and Gym/Gymnasium compatibility.

This module provides common utilities used by local and distributed trainers,
including deterministic seeding, environment-step normalization, reset-result
normalization, reset-mode selection, and policy evaluation.
"""

import random

import gym
import numpy as np
import torch
from tqdm import tqdm


def seed_all(seed):
    """Seed Python, NumPy, and PyTorch random number generators.

    Parameters
    ----------
    seed : int or None
        Random seed. If ``None``, the seed is replaced with ``0``.

    Returns
    -------
    None
        Global random number generators and PyTorch deterministic backend
        options are configured in place.
    """

    if seed is None:
        seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_next_step(env, action):
    """Step an environment and normalize Gym/Gymnasium return values.

    For continuous ``gym.spaces.Box`` action spaces, the function assumes that
    ``action`` is normalized to ``[-1, 1]`` and rescales it to the environment
    action bounds before calling ``env.step``. For non-Box action spaces, the
    action is forwarded unchanged.

    Parameters
    ----------
    env : gym.Env
        Environment to step.
    action : array-like or int
        Action selected by the agent. Continuous actions are expected to be in
        normalized ``[-1, 1]`` coordinates.

    Returns
    -------
    next_state : object
        Next observation returned by the environment.
    reward : float
        Reward returned by the environment.
    terminated : bool
        Whether the episode reached a terminal state.
    truncated : bool
        Whether the episode was truncated by a time limit or external
        constraint.
    info : dict
        Auxiliary environment information.

    Raises
    ------
    ValueError
        If ``env.step`` does not return either a Gym-style 4-tuple or a
        Gymnasium-style 5-tuple.
    """

    if isinstance(env.action_space, gym.spaces.Box):
        max_action = env.action_space.high
        min_action = env.action_space.low
        x = 0.5 * (action + 1) * (max_action - min_action) + min_action
        step_result = env.step(x)
    else:
        step_result = env.step(action)

    if isinstance(step_result, tuple) and len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
    elif isinstance(step_result, tuple) and len(step_result) == 4:
        next_state, reward, done, info = step_result
        terminated, truncated = done, False
    else:
        raise ValueError("env.step must return a 4-tuple or 5-tuple.")

    return next_state, reward, terminated, truncated, info


def get_reset_state(env, *args, **kwargs):
    """Reset an environment and return only the initial state.

    This helper supports both Gym-style resets that return only an observation
    and Gymnasium-style resets that return ``(observation, info)``.

    Parameters
    ----------
    env : gym.Env
        Environment to reset.
    *args : tuple
        Positional arguments forwarded to ``env.reset``.
    **kwargs : dict[str, object]
        Keyword arguments forwarded to ``env.reset``.

    Returns
    -------
    object
        Initial observation returned by the environment.
    """

    reset_result = env.reset(*args, **kwargs)

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0]

    return reset_result


def get_reset_result(env, *args, **kwargs):
    """Reset an environment and return ``(state, info)``.

    This helper supports both Gym-style resets that return only an observation
    and Gymnasium-style resets that return ``(observation, info)``. For
    Gym-style resets, an empty info dictionary is supplied.

    Parameters
    ----------
    env : gym.Env
        Environment to reset.
    *args : tuple
        Positional arguments forwarded to ``env.reset``.
    **kwargs : dict[str, object]
        Keyword arguments forwarded to ``env.reset``.

    Returns
    -------
    state : object
        Initial observation returned by the environment.
    info : dict
        Reset metadata. Empty when the environment returns only an observation.
    """

    reset_result = env.reset(*args, **kwargs)

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result

    return reset_result, {}


def select_reset_mode(env, info, random_reset_probability=0.01):
    """Select the reset mode after an episode finishes.

    Non-convergent simulator transitions force a random reset. Otherwise, the
    function uses a small random-reset probability for exploration and prefers
    ``"continue_last"`` when the environment exposes a current state.

    Parameters
    ----------
    env : gym.Env
        Environment being reset.
    info : dict
        Step information dictionary returned by the environment.
    random_reset_probability : float, optional
        Probability of forcing a random reset even when continuation is
        possible. The default is ``0.01``.

    Returns
    -------
    str
        Selected reset mode, either ``"random"`` or ``"continue_last"``.
    """

    if isinstance(info, dict) and info.get("is_non_convergent"):
        return "random"

    if random.random() < float(random_reset_probability):
        return "random"

    if hasattr(env, "state"):
        return "continue_last"

    return "random"


def evaluate(env, agent, seed, eval_iters):
    """Evaluate an agent for multiple episodes.

    Parameters
    ----------
    env : gym.Env
        Evaluation environment.
    agent : object
        Agent exposing an ``act`` method. The method is called as
        ``agent.act(state, training=False)``.
    seed : int or None
        Base seed for evaluation resets. If ``None``, the seed is replaced
        with ``0``.
    eval_iters : int
        Number of evaluation episodes to run.

    Returns
    -------
    total_ep_ret : list[float]
        Episode returns collected during evaluation.
    total_ep_len : list[int]
        Episode lengths collected during evaluation.
    """

    if seed is None:
        seed = 0

    total_ep_ret, total_ep_len = [], []

    for _ in tqdm(range(eval_iters), desc="EVALUATION"):
        ep_ret, ep_len, finished = 0.0, 0, False
        rand_seed = seed + np.random.randint(0, 1000)

        state = get_reset_state(
            env,
            seed=rand_seed,
            options={"reset_mode": "random"},
        )

        while not finished:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = get_next_step(
                env,
                action,
            )

            ep_ret += reward
            ep_len += 1
            state = next_state
            finished = terminated or truncated

        total_ep_ret.append(ep_ret)
        total_ep_len.append(ep_len)

    return total_ep_ret, total_ep_len