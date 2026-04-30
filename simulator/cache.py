"""In-memory LRU cache for simulation results.

This module defines a lightweight least-recently-used cache for simulator
outputs. It stores performance vectors and stability factors keyed by
simulation configuration, and tracks basic cache statistics.
"""

from collections import OrderedDict

import numpy as np


class SimulationCache:
    """Small LRU cache for simulator outputs.

    Parameters
    ----------
    enabled : bool, optional
        Whether cache lookup and insertion are active. If ``False``,
        :meth:`get` always returns ``None`` and :meth:`put` does nothing.
        The default is ``True``.
    maxsize : int, optional
        Maximum number of cached entries. When the cache exceeds this size,
        the least-recently-used entry is evicted. The default is ``256``.

    Attributes
    ----------
    enabled : bool
        Whether the cache is active.
    maxsize : int
        Maximum number of retained cache entries.
    _cache : collections.OrderedDict
        Ordered mapping from cache keys to cached simulation results.
    _stats : dict[str, int]
        Cache statistics containing ``"hits"``, ``"misses"``, and
        ``"evictions"``.
    """

    def __init__(self, *, enabled=True, maxsize=256):
        """Initialize the simulation cache.

        Parameters
        ----------
        enabled : bool, optional
            Whether the cache is active. The default is ``True``.
        maxsize : int, optional
            Maximum number of retained cache entries. The default is ``256``.

        Returns
        -------
        None
            Cache storage and statistics are initialized in place.
        """

        self.enabled = bool(enabled)
        self.maxsize = int(maxsize)
        self._cache = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @property
    def storage(self):
        """Return the underlying ordered cache mapping.

        Returns
        -------
        collections.OrderedDict
            Ordered mapping from cache keys to cached
            ``(performances, stability_factor)`` pairs.
        """

        return self._cache

    @property
    def stats(self):
        """Return cache hit, miss, and eviction counters.

        Returns
        -------
        dict[str, int]
            Copy of the cache statistics dictionary.
        """

        return dict(self._stats)

    def reset(self):
        """Clear all cached entries and reset counters.

        Returns
        -------
        None
            Cache storage is emptied and statistics are reset in place.
        """

        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key):
        """Return a cached result and update LRU state on hit.

        Parameters
        ----------
        key : hashable
            Cache key identifying a simulation configuration.

        Returns
        -------
        tuple[numpy.ndarray, float] or None
            Cached ``(performances, stability_factor)`` pair if present and
            caching is enabled; otherwise ``None``.
        """

        if not self.enabled:
            return None

        cached = self._cache.get(key)

        if cached is None:
            self._stats["misses"] += 1
            return None

        self._cache.move_to_end(key)
        self._stats["hits"] += 1

        return cached

    def put(self, key, performances, stability_factor):
        """Insert or update one cached simulation result.

        Parameters
        ----------
        key : hashable
            Cache key identifying a simulation configuration.
        performances : array-like or numpy.ndarray
            Performance vector to cache. A copy is stored to avoid accidental
            external mutation.
        stability_factor : float
            Stability factor associated with the cached performance vector.

        Returns
        -------
        None
            Cache storage is updated in place. If the cache exceeds
            ``maxsize``, least-recently-used entries are evicted.
        """

        if not self.enabled:
            return

        self._cache[key] = (
            np.array(performances, copy=True),
            float(stability_factor),
        )
        self._cache.move_to_end(key)

        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1