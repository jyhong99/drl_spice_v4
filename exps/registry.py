"""Agent registry used by experiment launch helpers."""

from agents.ddpg import DDPG
from agents.ppo import PPO
from agents.sac import SAC
from agents.td3 import TD3


def get_agent_registry():
    """Return supported agent configurations for experiment launching.

    Returns
    -------
    dict[str, tuple[type, bool, str]]
        Mapping of agent key to ``(agent_class, prioritized_mode, suffix)``.
    """

    registry = {
        "ppo": (PPO, False, "default"),
        "ddpg": (DDPG, False, "default"),
        "td3": (TD3, False, "default"),
        "sac": (SAC, False, "default"),
        "ddpg_per": (DDPG, True, "prior"),
        "td3_per": (TD3, True, "prior"),
        "sac_per": (SAC, True, "prior"),
        "random": (DDPG, True, "default"),
    }
    return registry
