"""Replay-buffer implementations and supporting data structures.

This package-level module re-exports replay-buffer classes, shared-buffer
classes, segment-tree data structures, storage utilities, and transition type
aliases. It provides a single import location for buffer components used by
training scripts, distributed runtimes, and agent implementations.

Examples
--------
Import common buffer classes directly from the package:

>>> from buffers import ReplayBuffer, PrioritizedReplayBuffer
>>> from buffers import SharedReplayBuffer, SumSegmentTree
"""

from buffers.local import BaseBuffer, RolloutBuffer, ReplayBuffer
from buffers.prioritized import PrioritizedReplayBuffer, SharedPrioritizedReplayBuffer
from buffers.segment_tree import SegmentTree, SumSegmentTree, MinSegmentTree
from buffers.shared import (
    SharedRolloutBuffer,
    SharedReplayBuffer,
)
from buffers.storage import (
    combined_shape,
    BufferStatsMixin,
    PicklePersistenceMixin,
    SharedBufferBase,
)
from buffers.types import (
    PrioritizedTransition,
    PrioritizedTransitionBatch,
    Transition,
    TransitionBatch,
)


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "SegmentTree",
    "SumSegmentTree",
    "MinSegmentTree",
    "SharedRolloutBuffer",
    "SharedReplayBuffer",
    "SharedPrioritizedReplayBuffer",
    "combined_shape",
    "BufferStatsMixin",
    "PicklePersistenceMixin",
    "SharedBufferBase",
    "Transition",
    "PrioritizedTransition",
    "TransitionBatch",
    "PrioritizedTransitionBatch",
]
"""list[str]: Public symbols exported when using ``from buffers import *``.

The exported names are:

- ``BaseBuffer``: Base NumPy-backed in-process transition buffer.
- ``RolloutBuffer``: Local on-policy rollout buffer.
- ``ReplayBuffer``: Local uniform off-policy replay buffer.
- ``PrioritizedReplayBuffer``: Local prioritized replay buffer.
- ``SegmentTree``: Generic array-backed segment tree.
- ``SumSegmentTree``: Segment tree for range-sum queries.
- ``MinSegmentTree``: Segment tree for range-minimum queries.
- ``SharedRolloutBuffer``: Ray-remote rollout buffer.
- ``SharedReplayBuffer``: Ray-remote uniform replay buffer.
- ``SharedPrioritizedReplayBuffer``: Ray-remote prioritized replay buffer.
- ``combined_shape``: Helper for constructing storage-array shapes.
- ``BufferStatsMixin``: Mixin for buffer usage counters.
- ``PicklePersistenceMixin``: Mixin for pickle-based save/load helpers.
- ``SharedBufferBase``: Base mixin for Ray-remote buffers.
- ``Transition``: Standard transition tuple type alias.
- ``PrioritizedTransition``: Prioritized transition tuple type alias.
- ``TransitionBatch``: Sequence type alias for standard transitions.
- ``PrioritizedTransitionBatch``: Sequence type alias for prioritized
  transitions.
"""