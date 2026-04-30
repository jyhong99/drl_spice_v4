"""Reward computation primitives for LNA optimization environments.

This module defines the default reward strategy used by modular LNA
optimization environments. The reward combines feasibility violation,
RF performance quality, variance regularization, and potential-based reward
shaping terms.
"""

import numpy as np


class Reward:
    """Default reward strategy for LNA optimization environments.

    This class provides helper methods used by
    :class:`envs.lna.modular.LNAEnvBase` to compute normalized performance
    progress, feasibility violation, RF figure of merit, variance penalty, and
    the final scalar reward.

    The methods expect the environment to provide attributes such as
    ``references``, ``target_spec``, ``p``, ``beta``, ``gamma``,
    ``penal_perf``, ``penal_viol``, ``lmda_viol``, ``lmda_perf``, and
    ``lmda_var``.
    """

    def make_f(self, env, performances):
        """Normalize raw performances between references and target specs.

        Parameters
        ----------
        env : LNAEnvBase
            Environment instance containing ``references`` and
            ``target_spec`` arrays.
        performances : array-like or numpy.ndarray
            Raw performance vector returned by the simulator.

        Returns
        -------
        numpy.ndarray
            Normalized performance progress vector with the same length as
            ``performances``.
        """

        return (performances - env.references) / (env.target_spec - env.references)

    def make_viol(self, env, performances=None, **kwargs):
        """Compute the aggregated feasibility-violation term.

        The violation is computed from normalized performance progress values.
        Components below the target threshold contribute positive violation,
        while components that already satisfy the target contribute zero.

        Parameters
        ----------
        env : LNAEnvBase
            Environment instance containing reward and feasibility attributes.
        performances : array-like or numpy.ndarray or None, optional
            Performance vector used to compute normalized progress. The default
            is ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments kept for interface compatibility.

        Returns
        -------
        float
            Aggregated violation value. A value of ``0.0`` indicates that all
            normalized performance components satisfy the target threshold.
        """

        f = self.make_f(env, performances)
        w = np.ones_like(f, dtype=float)

        viol = np.maximum(0.0, 1.0 - f)
        viol_norm = (np.dot(w, viol ** env.p) / np.sum(w)) ** (1.0 / env.p)

        env.is_feasible = viol_norm <= 0.0
        return viol_norm

    def make_perf(self, env, performances=None, eps=1e-9, **kwargs):
        """Compute the performance objective and RF figure of merit.

        The method supports two performance-vector formats:

        - ``[S11, S21, S22, NF, PD, IIP3]``
        - ``[S11, S21, S22, NF, PD]``

        S-parameters and noise figure are assumed to be in dB, power
        dissipation is assumed to be in mW, and IIP3 is assumed to be in dBm.
        If IIP3 is omitted, it is treated as ``0.0`` dBm.

        Parameters
        ----------
        env : LNAEnvBase
            Environment instance. It is accepted for interface consistency,
            but this method mainly uses ``performances``.
        performances : array-like or numpy.ndarray or None, optional
            Raw performance vector returned by the simulator. The default is
            ``None``.
        eps : float, optional
            Small positive constant used to avoid division by zero and
            logarithms of zero. The default is ``1e-9``.
        **kwargs : dict[str, object]
            Additional keyword arguments kept for interface compatibility.

        Returns
        -------
        perf : float
            Scalar performance objective used by the reward function. Lower
            values are better because this implementation returns
            ``-fom_db``.
        fom : float
            Linear RF figure-of-merit value.

        Raises
        ------
        ValueError
            If ``performances`` does not contain either five or six elements.
        """

        if len(performances) == 6:
            s11_db, s21_db, s22_db, nf_db, pd_mw, iip3_dbm = performances
        elif len(performances) == 5:
            s11_db, s21_db, s22_db, nf_db, pd_mw = performances
            iip3_dbm = 0.0
        else:
            raise ValueError(
                f"Unexpected performance vector length: {len(performances)}"
            )

        s11 = 10 ** (s11_db / 20.0)
        s21 = 10 ** (s21_db / 20.0)
        s22 = 10 ** (s22_db / 20.0)
        nf = 10 ** (nf_db / 10.0)
        iip3_mw = 10 ** (iip3_dbm / 10.0)

        in_match = max(1.0 - s11, eps)
        out_match = max(1.0 - s22, eps)
        noise_factor = nf - 1.0

        fom = (
            in_match
            * s21
            * out_match
            * iip3_mw
        ) / (noise_factor * pd_mw + eps)

        fom_db = (
            20.0 * np.log10(in_match)
            + s21_db
            + 20.0 * np.log10(out_match)
            - 10.0 * np.log10(noise_factor + eps)
            - 10.0 * np.log10(pd_mw + eps)
            + iip3_dbm
        )

        return -fom_db, fom

    def make_var(self, env, performances=None, **kwargs):
        """Compute variance of normalized performance components.

        Parameters
        ----------
        env : LNAEnvBase
            Environment instance containing normalization references and
            targets.
        performances : array-like or numpy.ndarray or None, optional
            Performance vector used to compute normalized progress. The default
            is ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments kept for interface compatibility.

        Returns
        -------
        float
            Variance of the normalized performance progress vector.
        """

        f = self.make_f(env, performances)
        return np.var(f)

    def make_reward(self, env, performances):
        """Compose the final scalar reward from reward components.

        This method computes the performance objective, feasibility violation,
        and variance term, applies invalid-design penalties when needed, and
        combines potential-based shaping terms into the final scalar reward.

        The method also updates reward-related environment attributes for
        logging and subsequent reward-shaping computations.

        Parameters
        ----------
        env : LNAEnvBase
            Environment instance whose reward-related attributes are updated
            in place.
        performances : array-like or numpy.ndarray
            Raw performance vector returned by the simulator.

        Returns
        -------
        float
            Final scalar reward for the current transition.
        """

        perf, fom = env._make_perf(performances)
        viol = env._make_viol(performances)
        var = env._make_var(performances)

        prev_perf = env.perf
        prev_viol = env.viol

        if env.is_invalid:
            perf += env.penal_perf
            viol += env.penal_viol

        pbrs_perf = -env.gamma * perf + prev_perf
        pbrs_viol = -env.gamma * viol + prev_viol

        reward_perf = -env.beta * perf + (1.0 - env.beta) * pbrs_perf
        reward_viol = -env.beta * viol + (1.0 - env.beta) * pbrs_viol
        reward_var = -var

        env.fom = fom
        env.perf = perf
        env.viol = viol
        env.var = var

        env.pbrs_perf = pbrs_perf
        env.pbrs_viol = pbrs_viol

        env.reward_perf = reward_perf
        env.reward_viol = reward_viol
        env.reward_var = reward_var

        env.reward = (
            env.lmda_viol * env.reward_viol
            + env.lmda_perf * env.reward_perf
            + env.lmda_var * env.reward_var
        )

        return env.reward