"""Progress-bar helpers for local and distributed training.

This module defines tqdm-based progress displays for training loops and a
Ray actor used to aggregate per-worker progress counts asynchronously during
distributed rollout collection.
"""

import math

import ray
from tqdm import tqdm


def _format_metric(value):
    """Format a numeric metric for compact progress-bar display.

    Parameters
    ----------
    value : int, float, or None
        Metric value to format. ``None`` and ``NaN`` values are displayed as
        ``"--"``.

    Returns
    -------
    str
        Compact string representation of the metric.
    """

    if value is None:
        return "--"

    try:
        if math.isnan(float(value)):
            return "--"
    except (TypeError, ValueError):
        return "--"

    return f"{float(value):.4g}"


class TrainingProgressBars:
    """Manage tqdm progress bars for training and worker progress.

    This class displays a global training progress bar and, when multiple
    runners are used, separate per-worker progress bars. It also tracks
    episode counts, latest returns, and best FOM values for progress-bar
    postfix display.

    Parameters
    ----------
    max_iters : int
        Maximum total number of training timesteps.
    n_runners : int
        Number of rollout runners. If greater than one, worker-specific
        progress bars are created.
    runner_iters : int, optional
        Maximum number of timesteps scheduled per runner task. The value is
        stored for display configuration. The default is ``1``.

    Attributes
    ----------
    max_iters : int
        Maximum total number of training timesteps.
    n_runners : int
        Number of rollout runners.
    runner_iters : int
        Maximum scheduled timesteps per runner task.
    _global_pbar : tqdm.tqdm or None
        Global training progress bar.
    _worker_pbars : list[tqdm.tqdm]
        Per-worker progress bars.
    _worker_pbar_map : dict[int, tqdm.tqdm]
        Mapping from runner index to progress bar.
    _worker_counts : dict[int, int]
        Current per-worker step counts.
    _worker_ep_counts : dict[int, int]
        Current per-worker completed-episode counts.
    _worker_last_rets : dict[int, float]
        Latest episode return observed for each worker.
    _worker_best_perfs : dict[int, float]
        Best FOM observed for each worker.
    """

    def __init__(self, *, max_iters, n_runners, runner_iters=1):
        """Initialize progress-bar bookkeeping.

        Parameters
        ----------
        max_iters : int
            Maximum total number of training timesteps.
        n_runners : int
            Number of rollout runners.
        runner_iters : int, optional
            Maximum scheduled timesteps per runner task. The default is ``1``.

        Returns
        -------
        None
            Progress-bar state containers are initialized in place.
        """

        self.max_iters = int(max_iters)
        self.n_runners = int(n_runners)
        self.runner_iters = max(1, int(runner_iters))

        self._global_pbar = None
        self._worker_pbars = []
        self._worker_pbar_map = {}
        self._worker_counts = {}
        self._worker_ep_counts = {}
        self._worker_last_rets = {}
        self._worker_best_perfs = {}

        self._section_pbars = []
        self._footer_pbar = None
        self._msg_pbar = None

    def setup(self):
        """Create and display the configured progress bars.

        Existing progress bars are closed before new bars are created. For
        distributed training, this method creates section headers,
        per-worker bars, a global training bar, and a footer/message line.
        For single-runner training, only the global training bar and message
        line are created.

        Returns
        -------
        None
            Progress bars are initialized and displayed in place.
        """

        self.close()

        msg_position = (self.n_runners + 4) if (self.n_runners > 1) else 0
        self._msg_pbar = tqdm(
            total=1,
            initial=0,
            position=msg_position,
            leave=True,
            bar_format="{desc}",
            dynamic_ncols=True,
            disable=False,
        )
        self._msg_pbar.set_description_str(" ", refresh=True)

        if self.n_runners > 1:
            worker_header = tqdm(
                total=0,
                initial=0,
                desc=f"{' WORKER PROGRESS ':=^96}",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}",
                disable=False,
            )
            self._section_pbars.append(worker_header)

            worker_total = int(
                math.ceil(float(self.max_iters) / float(max(1, self.n_runners)))
            )

            for runner_idx in range(self.n_runners):
                pbar = tqdm(
                    total=worker_total,
                    initial=0,
                    desc=f"WORKER {runner_idx + 1:02d}",
                    unit="step",
                    position=1 + runner_idx,
                    leave=True,
                    dynamic_ncols=True,
                    disable=False,
                )
                self._worker_pbars.append(pbar)
                self._worker_pbar_map[runner_idx] = pbar
                self._worker_counts[runner_idx] = 0
                self._worker_ep_counts[runner_idx] = 0
                self._worker_last_rets[runner_idx] = None
                self._worker_best_perfs[runner_idx] = None

            training_header = tqdm(
                total=0,
                initial=0,
                desc=f"{' TRAINING PROGRESS ':=^96}",
                position=self.n_runners + 1,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}",
                disable=False,
            )
            self._section_pbars.append(training_header)

            global_desc = "TRAINING"
            global_position = self.n_runners + 2

            self._footer_pbar = tqdm(
                total=0,
                initial=0,
                desc="=" * 96,
                position=self.n_runners + 3,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}",
                disable=False,
            )
        else:
            global_desc = "TRAINING"
            global_position = 1

        self._global_pbar = tqdm(
            total=self.max_iters,
            initial=0,
            desc=global_desc,
            unit="step",
            position=global_position,
            leave=True,
            dynamic_ncols=True,
            disable=False,
        )

    def update_worker(self, runner_name, step_inc=1):
        """Advance one worker progress bar.

        Parameters
        ----------
        runner_name : int or str
            Runner identifier associated with a worker progress bar.
        step_inc : int, optional
            Number of steps to add to the worker progress count. Non-positive
            increments are ignored. The default is ``1``.

        Returns
        -------
        None
            The worker progress bar and internal count are updated in place.
        """

        pbar = self._worker_pbar_map.get(int(runner_name))

        if pbar is None:
            return

        delta = int(step_inc)

        if delta > 0:
            pbar.update(delta)
            self._worker_counts[int(runner_name)] = (
                self._worker_counts.get(int(runner_name), 0) + delta
            )

    def sync_worker_counts(self, counts):
        """Synchronize worker bars to externally reported step counts.

        Parameters
        ----------
        counts : dict[int or str, int] or None
            Mapping from runner identifiers to target step counts.

        Returns
        -------
        None
            Worker progress bars are advanced to match the reported counts.
        """

        for runner_name, target_count in (counts or {}).items():
            runner_idx = int(runner_name)
            current_count = int(self._worker_counts.get(runner_idx, 0))
            delta = int(target_count) - current_count

            if delta > 0:
                self.update_worker(runner_idx, delta)

    def update_worker_postfix(self, runner_name, *, ep, ret, best_fom):
        """Update a worker progress-bar postfix.

        Parameters
        ----------
        runner_name : int or str
            Runner identifier associated with a worker progress bar.
        ep : int or None
            Completed episode count displayed for the worker.
        ret : float or None
            Latest episode return displayed for the worker.
        best_fom : float or None
            Best FOM displayed for the worker.

        Returns
        -------
        None
            The worker progress-bar postfix is updated when the bar exists.
        """

        pbar = self._worker_pbar_map.get(int(runner_name))

        if pbar is None:
            return

        pbar.set_postfix(
            {
                "EP": ep if ep is not None else "--",
                "RET": _format_metric(ret),
                "BEST FOM": _format_metric(best_fom),
            },
            refresh=False,
        )

    def update_worker_stats(self, runner_name, *, ep_ret=None, best_fom=None):
        """Update worker statistics and refresh its progress-bar postfix.

        Parameters
        ----------
        runner_name : int or str
            Runner identifier associated with a worker progress bar.
        ep_ret : sequence[float] or None, optional
            Episode returns completed by this worker. The latest value is used
            for display, and the number of returns increments the episode
            count. The default is ``None``.
        best_fom : float or None, optional
            Best FOM observed by this worker in the latest rollout chunk. The
            displayed best FOM is updated monotonically. The default is
            ``None``.

        Returns
        -------
        None
            Worker statistics and postfix values are updated in place.
        """

        runner_idx = int(runner_name)
        returns = list(ep_ret or [])

        if returns:
            self._worker_ep_counts[runner_idx] = (
                self._worker_ep_counts.get(runner_idx, 0) + len(returns)
            )
            self._worker_last_rets[runner_idx] = float(returns[-1])

        current_best = self._worker_best_perfs.get(runner_idx)

        if best_fom is not None:
            current_best = (
                best_fom
                if current_best is None
                else max(float(current_best), float(best_fom))
            )
            self._worker_best_perfs[runner_idx] = current_best

        ep_count = self._worker_ep_counts.get(runner_idx, 0)

        self.update_worker_postfix(
            runner_idx,
            ep=ep_count,
            ret=self._worker_last_rets.get(runner_idx),
            best_fom=self._worker_best_perfs.get(runner_idx),
        )

    def _global_progress_count(self, timesteps):
        """Compute the target global progress count.

        Parameters
        ----------
        timesteps : int
            Scheduler-reported global timestep count.

        Returns
        -------
        int
            Target global progress count. In multi-runner mode, this is at
            least the sum of the worker progress counts.
        """

        target = int(timesteps)

        if self.n_runners > 1:
            target = max(target, int(sum(self._worker_counts.values())))

        return target

    def refresh_global_from_workers(self, *, timesteps=0):
        """Refresh the global progress bar from worker counts.

        Parameters
        ----------
        timesteps : int, optional
            Scheduler-reported global timestep count. The default is ``0``.

        Returns
        -------
        None
            The global progress bar is advanced if the computed target count
            exceeds its current count.
        """

        if self._global_pbar is None:
            return

        delta = self._global_progress_count(timesteps) - int(self._global_pbar.n)

        if delta > 0:
            self._global_pbar.update(delta)

    def update_global(self, *, timesteps, ep, ret, best_fom=None):
        """Update the global training progress bar.

        Parameters
        ----------
        timesteps : int
            Current global timestep count.
        ep : int or None
            Global completed episode count.
        ret : float or None
            Latest global episode return.
        best_fom : float or None, optional
            Best FOM observed globally. The default is ``None``.

        Returns
        -------
        None
            The global progress count and postfix are updated in place.
        """

        if self._global_pbar is None:
            return

        self.refresh_global_from_workers(timesteps=timesteps)
        self._global_pbar.set_postfix(
            {
                "EP": ep if ep is not None else "--",
                "RET": _format_metric(ret),
                "BEST FOM": _format_metric(best_fom),
            },
            refresh=False,
        )

    def close(self):
        """Close all active progress bars and reset internal state.

        Returns
        -------
        None
            All tqdm progress bars are closed. Internal progress counters and
            mappings are cleared.
        """

        try:
            if self._global_pbar is not None:
                self._global_pbar.close()
        except Exception:
            pass
        self._global_pbar = None

        try:
            for pbar in self._worker_pbars:
                pbar.close()
        except Exception:
            pass

        self._worker_pbars = []
        self._worker_pbar_map = {}
        self._worker_counts = {}
        self._worker_ep_counts = {}
        self._worker_last_rets = {}
        self._worker_best_perfs = {}

        try:
            for pbar in self._section_pbars:
                pbar.close()
        except Exception:
            pass

        self._section_pbars = []

        try:
            if self._footer_pbar is not None:
                self._footer_pbar.close()
        except Exception:
            pass

        self._footer_pbar = None

        try:
            if self._msg_pbar is not None:
                self._msg_pbar.close()
        except Exception:
            pass

        self._msg_pbar = None


@ray.remote(num_gpus=0)
class WorkerProgressTracker:
    """Ray actor that accumulates per-worker progress counts.

    Parameters
    ----------
    n_runners : int
        Number of runner indices initialized in the counter dictionary.

    Attributes
    ----------
    counts : dict[int, int]
        Mapping from runner index to reported step count.
    """

    def __init__(self, n_runners):
        """Initialize worker progress counters.

        Parameters
        ----------
        n_runners : int
            Number of worker counters to initialize.

        Returns
        -------
        None
            Worker progress counters are initialized in place.
        """

        self.counts = {
            runner_idx: 0
            for runner_idx in range(int(n_runners))
        }

    def increment(self, runner_name, delta=1):
        """Increment one worker progress counter.

        Parameters
        ----------
        runner_name : int or str
            Runner identifier whose counter should be incremented.
        delta : int, optional
            Increment amount. The default is ``1``.

        Returns
        -------
        None
            The selected worker counter is incremented in place.
        """

        runner_idx = int(runner_name)
        self.counts[runner_idx] = self.counts.get(runner_idx, 0) + int(delta)

    def snapshot(self):
        """Return a copy of current worker progress counts.

        Returns
        -------
        dict[int, int]
            Copy of the current per-worker progress counts.
        """

        return dict(self.counts)