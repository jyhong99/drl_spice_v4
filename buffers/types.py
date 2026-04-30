"""Shared transition type aliases for buffer interfaces.

This module defines common type aliases used by replay-buffer and rollout
buffer interfaces. The aliases make function signatures easier to read and
help document the expected structure of transition tuples passed between
workers, buffers, and training components.

A standard transition is represented as:

``(state, action, reward, next_state, terminated, truncated)``

A prioritized transition additionally includes an explicit priority value:

``(state, action, reward, next_state, terminated, truncated, priority)``
"""

from typing import Any, Sequence, Tuple


Transition = Tuple[Any, Any, Any, Any, Any, Any]
"""tuple[Any, Any, Any, Any, Any, Any]: Standard transition tuple.

The tuple fields are:

- ``state``: Observation before applying the action.
- ``action``: Action taken in the environment.
- ``reward``: Reward observed after the transition.
- ``next_state``: Observation reached after applying the action.
- ``terminated``: Terminal flag.
- ``truncated``: Truncation flag.
"""


PrioritizedTransition = Tuple[Any, Any, Any, Any, Any, Any, Any]
"""tuple[Any, Any, Any, Any, Any, Any, Any]: Prioritized transition tuple.

The tuple fields are:

- ``state``: Observation before applying the action.
- ``action``: Action taken in the environment.
- ``reward``: Reward observed after the transition.
- ``next_state``: Observation reached after applying the action.
- ``terminated``: Terminal flag.
- ``truncated``: Truncation flag.
- ``priority``: Sampling priority assigned to the transition.
"""


TransitionBatch = Sequence[Transition]
"""Sequence[Transition]: Batch or iterable sequence of standard transitions."""


PrioritizedTransitionBatch = Sequence[PrioritizedTransition]
"""Sequence[PrioritizedTransition]: Batch or iterable sequence of prioritized transitions."""


__all__ = [
    "Transition",
    "PrioritizedTransition",
    "TransitionBatch",
    "PrioritizedTransitionBatch",
]
"""list[str]: Public symbols exported when using ``from buffers.types import *``."""