"""Agent implementations exposed by the training stack.

This package-level module re-exports the reinforcement-learning agent
classes implemented in the ``agents`` package. It provides a convenient
single import location for training scripts, experiment runners, and runtime
components.

Examples
--------
Import agent classes directly from the package:

>>> from agents import DDPG, TD3, PPO, SAC
"""

from agents.ddpg import DDPG
from agents.ppo import PPO
from agents.sac import SAC
from agents.td3 import TD3

__all__ = ["DDPG", "TD3", "PPO", "SAC"]
"""list[str]: Public symbols exported when using ``from agents import *``.

The exported names are:

- ``DDPG``: Deep Deterministic Policy Gradient agent.
- ``TD3``: Twin Delayed Deep Deterministic Policy Gradient agent.
- ``PPO``: Proximal Policy Optimization agent.
- ``SAC``: Soft Actor-Critic agent.
"""