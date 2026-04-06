"""Pairwise Response Model: hybrid backbone + counterfactual pairwise replay.

Architecture:
  Score(x, ℓ) = μ_hier(ℓ, b) + r_ctx(x, ℓ) + r_pw(x, ℓ)

Pairwise training:
  For each teaching step, construct counterfactual pairwise labels
  from short surrogate rollouts of top-K candidate lessons.
  Train Bradley-Terry: P(ℓ_i ≻ ℓ_j | x) = σ(s_i - s_j)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .lesson_response_model_v3 import LessonResponseModelV3, mastery_bucket, PROBE_NAMES

N_LESSONS = 13  # Must match len(LESSON_CATALOG_V2)


def _build_phi(lesson_name: str, catalog_names: list,
               theta: str, mastery: dict,
               nu: float, tau: float, gg: float, gs: float, kappa: float,
               budget: float, recent: dict,
               severity: float = 0.5, novelty: float = 0.3) -> np.ndarray:
    one_hot = np.zeros(N_LESSONS)
    if lesson_name in catalog_names:
        one_hot[catalog_names.index(lesson_name)] = 1.0
    theta_enc = np.array([1.0, 0.0] if theta == "safe" else [0.0, 1.0])
    u = np.array([mastery.get(p, 0.5) for p in PROBE_NAMES])
    state = np.array([kappa, tau, nu, gs, gg])
    bud = np.array([min(budget / 4.0, 2.0)])
    rec = np.array([recent.get(lesson_name, 0) / 3.0])
    ep = np.array([severity, novelty])
    return np.concatenate([one_hot, theta_enc, u, state, bud, rec, ep])


@dataclass
class BayesLinHead:
    """Bayesian linear head: y = w'φ + ε."""
    prior_precision: float = 2.0
    noise_var: float = 0.15
    _A: np.ndarray = None
    _b: np.ndarray = None
    _n: int = 0

    def _init(self, d):
        if self._A is None:
            self._A = self.prior_precision * np.eye(d)
            self._b = np.zeros(d)

    def update(self, phi: np.ndarray, y: float):
        self._init(len(phi))
        self._A += np.outer(phi, phi) / self.noise_var
        self._b += phi * y / self.noise_var
        self._n += 1

    def predict(self, phi: np.ndarray) -> tuple[float, float]:
        self._init(len(phi))
        try: S = np.linalg.inv(self._A)
        except np.linalg.LinAlgError: S = np.eye(len(phi)) / self.prior_precision
        mu = S @ self._b
        return float(phi @ mu), float(max(phi @ S @ phi + self.noise_var, 1e-8))


@dataclass
class PairwiseHead:
    """Bradley-Terry pairwise: P(i≻j) = σ(w'(φ_i - φ_j))."""
    prior_precision: float = 0.5
    noise_var: float = 0.2
    _A: np.ndarray = None
    _b: np.ndarray = None
    _n: int = 0

    def _init(self, d):
        if self._A is None:
            self._A = self.prior_precision * np.eye(d)
            self._b = np.zeros(d)

    def update_pair(self, phi_i: np.ndarray, phi_j: np.ndarray, y_ij: float):
        """y_ij in [-1, 1]: positive means i better than j."""
        dphi = phi_i - phi_j
        self._init(len(dphi))
        self._A += np.outer(dphi, dphi) / self.noise_var
        self._b += dphi * y_ij / self.noise_var
        self._n += 1

    def score(self, phi: np.ndarray) -> tuple[float, float]:
        """Score a single lesson for ranking: w'φ."""
        self._init(len(phi))
        try: S = np.linalg.inv(self._A)
        except np.linalg.LinAlgError: S = np.eye(len(phi)) / self.prior_precision
        mu = S @ self._b
        return float(phi @ mu), float(max(phi @ S @ phi, 1e-8))


@dataclass
class PairwiseResponseModel:
    """Hybrid + pairwise counterfactual response model."""
    hier: LessonResponseModelV3 = None
    catalog_names: list = field(default_factory=list)
    theta: str = "safe"

    # Contextual residual
    res_gain: BayesLinHead = field(default_factory=lambda: BayesLinHead(prior_precision=2.0, noise_var=0.15))
    res_harm: BayesLinHead = field(default_factory=lambda: BayesLinHead(prior_precision=2.0, noise_var=0.2))

    # Pairwise heads (gain and harm)
    pw_gain: PairwiseHead = field(default_factory=lambda: PairwiseHead(prior_precision=0.5, noise_var=0.2))
    pw_harm: PairwiseHead = field(default_factory=lambda: PairwiseHead(prior_precision=0.5, noise_var=0.25))

    # Observation buffer for counterfactual replay
    _obs_buffer: list = field(default_factory=list)
    _max_buffer: int = 50

    def __post_init__(self):
        if self.hier is None:
            self.hier = LessonResponseModelV3()

    def _phi(self, lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, sev=0.5, nov=0.3):
        return _build_phi(lesson_name, self.catalog_names, self.theta, mastery,
                          nu, tau, gg, gs, kappa, budget, recent, sev, nov)

    def predict_gain(self, lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, sev=0.5, nov=0.3):
        bucket = mastery_bucket(mastery)
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, sev, nov)
        # Hier backbone
        h_mean = sum(self.hier.gain_expected(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)
        h_var = sum(self.hier.gain_variance(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)
        # Residual
        r_mean, r_var = self.res_gain.predict(phi)
        # Pairwise score
        pw_mean, pw_var = self.pw_gain.score(phi)
        mean = h_mean + 0.5 * r_mean + 0.4 * pw_mean
        var = h_var + 0.25 * r_var + 0.16 * pw_var
        return {"mean": mean, "var": var, "phi": phi,
                "hier": h_mean, "res": r_mean, "pw": pw_mean}

    def predict_harm(self, lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, sev=0.5, nov=0.3):
        bucket = mastery_bucket(mastery)
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, sev, nov)
        h_harm = self.hier.total_harm(lesson_name, bucket) / 6.0  # normalized
        r_mean, r_var = self.res_harm.predict(phi)
        pw_mean, pw_var = self.pw_harm.score(phi)
        mean = h_harm + 0.3 * r_mean + 0.2 * pw_mean
        var = 0.09 * r_var + 0.04 * pw_var + 0.01
        return {"mean": mean, "var": var, "hier": h_harm, "res": r_mean, "pw": pw_mean}

    def update(self, lesson_name, mastery_before, mastery_after,
               nu, nu_after, tau, gg, gg_after, gs, kappa,
               budget, recent, sev, nov,
               otr_before, otr_after):
        bucket = mastery_bucket(mastery_before)
        phi = self._phi(lesson_name, mastery_before, nu, tau, gg, gs, kappa, budget, recent, sev, nov)

        # Hier backbone update
        self.hier.update_gain(lesson_name, bucket, mastery_before, mastery_after)
        self.hier.update_harm(lesson_name, bucket, nu, nu_after, gg, gg_after, otr_before, otr_after)

        # Gain/harm outcomes
        gain = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5) for p in PROBE_NAMES) / len(PROBE_NAMES)
        harm = (otr_after - otr_before) + 0.5 * (nu_after - nu) + 0.5 * (gg_after - gg)

        # Residual update (target = outcome - hier prediction)
        h_gain = sum(self.hier.gain_expected(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)
        h_harm = self.hier.total_harm(lesson_name, bucket) / 6.0
        self.res_gain.update(phi, gain - h_gain)
        self.res_harm.update(phi, harm - h_harm)

        # Store observation
        obs = {"phi": phi, "gain": gain, "harm": harm, "lesson": lesson_name}
        self._obs_buffer.append(obs)
        if len(self._obs_buffer) > self._max_buffer:
            self._obs_buffer.pop(0)

        # Counterfactual pairwise replay: compare with ALL different-lesson observations
        for prev in self._obs_buffer[:-1]:
            if prev["lesson"] != lesson_name:
                dg = gain - prev["gain"]
                dh = harm - prev["harm"]
                # Normalized pairwise label: clip to [-1, 1]
                y_gain = float(np.clip(dg * 5.0, -1, 1))
                y_harm = float(np.clip(dh * 5.0, -1, 1))
                self.pw_gain.update_pair(phi, prev["phi"], y_gain)
                self.pw_harm.update_pair(phi, prev["phi"], y_harm)

    def counterfactual_replay(self, state_phi: np.ndarray,
                              candidate_phis: list[np.ndarray],
                              candidate_gains: list[float],
                              candidate_harms: list[float]):
        """Inject synthetic pairwise data from surrogate rollouts."""
        n = len(candidate_phis)
        for i in range(n):
            for j in range(i + 1, n):
                dg = candidate_gains[i] - candidate_gains[j]
                dh = candidate_harms[i] - candidate_harms[j]
                y_g = float(np.clip(dg * 5.0, -1, 1))
                y_h = float(np.clip(dh * 5.0, -1, 1))
                self.pw_gain.update_pair(candidate_phis[i], candidate_phis[j], y_g)
                self.pw_harm.update_pair(candidate_phis[i], candidate_phis[j], y_h)

    def n_updated(self) -> int:
        return self.hier.n_updated() + self.res_gain._n + self.pw_gain._n
