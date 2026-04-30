"""Segment-tree data structures used by prioritized replay.

This module defines array-backed segment-tree implementations used by
prioritized experience replay. Segment trees allow efficient range queries
and point updates, which are useful for priority-based sampling and
importance-sampling weight normalization.

Classes
-------
SegmentTree
    Generic segment tree supporting associative range aggregation.
SumSegmentTree
    Segment tree specialized for range-sum queries and prefix-sum retrieval.
MinSegmentTree
    Segment tree specialized for range-minimum queries.
"""

import operator
from typing import Callable


class SegmentTree(object):
    """Array-backed segment tree supporting range aggregation.

    This class implements a generic segment tree where each internal node
    stores the result of applying an associative binary operation to its two
    child nodes. It supports point updates and range aggregation over leaf
    values.

    Parameters
    ----------
    capacity : int
        Number of leaf slots in the tree. In typical usage, this should be a
        positive integer, often a power of two.
    operation : Callable[[float, float], float]
        Associative binary operation used to combine child-node values.
        Examples include ``operator.add`` for sums and ``min`` for minimums.
    init_value : float
        Initial value stored in every node of the tree.

    Attributes
    ----------
    capacity : int
        Number of leaf slots in the tree.
    tree : list[float]
        Flat array representation of the segment tree. Leaf nodes are stored
        from index ``capacity`` to ``2 * capacity - 1``.
    operation : Callable[[float, float], float]
        Binary operation used for aggregation.
    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialize the segment tree.

        Parameters
        ----------
        capacity : int
            Number of leaf slots in the tree.
        operation : Callable[[float, float], float]
            Associative binary operation used for aggregation.
        init_value : float
            Initial value assigned to all tree nodes.

        Returns
        -------
        None
            The constructor initializes the tree storage in place.
        """

        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self,
        start: int,
        end: int,
        node: int,
        node_start: int,
        node_end: int,
    ) -> float:
        """Recursively aggregate values over an inclusive range.

        Parameters
        ----------
        start : int
            Inclusive start index of the query range.
        end : int
            Inclusive end index of the query range.
        node : int
            Current tree node index in the flat tree array.
        node_start : int
            Inclusive start index of the interval represented by ``node``.
        node_end : int
            Inclusive end index of the interval represented by ``node``.

        Returns
        -------
        float
            Aggregated value over the query interval ``[start, end]``.
        """

        if start == node_start and end == node_end:
            return self.tree[node]

        mid = (node_start + node_end) // 2

        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)

        if mid + 1 <= start:
            return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)

        return self.operation(
            self._operate_helper(start, mid, 2 * node, node_start, mid),
            self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
        )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Aggregate values over a leaf-index range.

        Parameters
        ----------
        start : int, optional
            Inclusive start index of the query range. The default is ``0``.
        end : int, optional
            Exclusive end index of the query range. If ``end <= 0``, it is
            interpreted relative to ``self.capacity``. For example,
            ``end=0`` means ``self.capacity`` and ``end=-1`` means
            ``self.capacity - 1``. The default is ``0``.

        Returns
        -------
        float
            Aggregated value over the range ``[start, end)`` after converting
            ``end`` to an inclusive internal index.
        """

        if end <= 0:
            end += self.capacity
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set a leaf value and update ancestor aggregates.

        Parameters
        ----------
        idx : int
            Leaf index to update.
        val : float
            New value to assign to the leaf.

        Returns
        -------
        None
            The selected leaf and all affected ancestor nodes are updated in
            place.
        """

        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(
                self.tree[2 * idx],
                self.tree[2 * idx + 1],
            )
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Return the value stored at a leaf index.

        Parameters
        ----------
        idx : int
            Leaf index to read.

        Returns
        -------
        float
            Value stored at the requested leaf.
        """

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """Segment tree specialized for summation queries.

    This tree stores numeric values and supports efficient range-sum queries.
    It also provides prefix-sum retrieval, which is useful for proportional
    sampling in prioritized replay.

    Parameters
    ----------
    capacity : int
        Number of leaf slots in the tree.
    """

    def __init__(self, capacity: int):
        """Initialize a sum segment tree.

        Parameters
        ----------
        capacity : int
            Number of leaf slots in the tree.

        Returns
        -------
        None
            The tree is initialized with zeros and summation as the aggregation
            operation.
        """

        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            init_value=0.0,
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Return the sum over a leaf-index range.

        Parameters
        ----------
        start : int, optional
            Inclusive start index of the query range. The default is ``0``.
        end : int, optional
            Exclusive end index of the query range. If ``end <= 0``, it is
            interpreted relative to ``self.capacity``. The default is ``0``.

        Returns
        -------
        float
            Sum of values in the range ``[start, end)``.
        """

        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the leaf index whose cumulative sum crosses a threshold.

        This method traverses the tree from the root to a leaf and returns the
        first index whose cumulative prefix sum is greater than ``upperbound``.
        It is commonly used for proportional sampling from priority masses.

        Parameters
        ----------
        upperbound : float
            Prefix-sum threshold. This value is typically sampled uniformly
            from ``[0, total_priority)``.

        Returns
        -------
        int
            Leaf index whose cumulative priority interval contains
            ``upperbound``.
        """

        idx = 1

        while idx < self.capacity:
            left = 2 * idx
            right = left + 1

            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = right

        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """Segment tree specialized for minimum queries.

    This tree stores numeric values and supports efficient range-minimum
    queries. In prioritized replay, it is commonly used to compute the minimum
    sampling probability for normalizing importance-sampling weights.

    Parameters
    ----------
    capacity : int
        Number of leaf slots in the tree.
    """

    def __init__(self, capacity: int):
        """Initialize a minimum segment tree.

        Parameters
        ----------
        capacity : int
            Number of leaf slots in the tree.

        Returns
        -------
        None
            The tree is initialized with infinity and minimum as the
            aggregation operation.
        """

        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            init_value=float("inf"),
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Return the minimum value over a leaf-index range.

        Parameters
        ----------
        start : int, optional
            Inclusive start index of the query range. The default is ``0``.
        end : int, optional
            Exclusive end index of the query range. If ``end <= 0``, it is
            interpreted relative to ``self.capacity``. The default is ``0``.

        Returns
        -------
        float
            Minimum value in the range ``[start, end)``.
        """

        return super(MinSegmentTree, self).operate(start, end)