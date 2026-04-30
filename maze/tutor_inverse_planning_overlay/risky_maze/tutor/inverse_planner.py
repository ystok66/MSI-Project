from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .candidates import generate_tutor_candidates
from .compat import get_any, make_tutor_action, normalize_action_name
from .context import TutorDecisionContext, coerce_context
from .diagnostics import TutorDecisionLog, TutorEpisodeDiagnostics
from .path_predictor import LearnerPathPredictor, PredictedPath, first_action_probability
from .profiles import LearnerProfile, default_profiles, likelihood_from_action_prob, normalize_belief, uniform_profile_belief
from .rollout import CounterfactualRolloutEvaluator, RolloutConfig, TutorActionValue
from .shadow import ShadowLearnerState, clone_from_snapshots
from .world_model import current_pos, remaining_time_from_state, step_count


@dataclass
class TutorConfig:
    mode: Literal["warning_only", "full"] = "full"
    top_k_paths: int = 4
    rollout_horizon: int = 10
    max_candidates: int = 20
    profile_belief_floor: float = 1e-4
    eval_disabled: bool = True
    waypoint_cooldown_steps: int = 4
    waypoint_min_advantage_over_wait: float = 0.10
    warning_min_advantage_over_wait: float = 0.00
    posterior_update_strength: float = 1.0
    rollout: RolloutConfig = field(default_factory=RolloutConfig)


class InversePlanningTutor:
    """Finite-profile posterior + short-horizon counterfactual tutor.

    This class is deliberately not a true nested I-POMDP solver.  It maintains a
    posterior over a finite set of learner profiles, builds shadow learners from
    snapshots, predicts top-K paths, and evaluates WAIT/WARNING/WAYPOINT through
    short rollouts.
    """

    def __init__(self, config: TutorConfig | None = None, profiles: list[LearnerProfile] | None = None):
        self.config = config or TutorConfig()
        self.profiles = profiles or default_profiles()
        self.profile_by_name = {p.name: p for p in self.profiles}
        self.profile_belief: dict[str, float] = uniform_profile_belief(self.profiles)
        self.path_predictor = LearnerPathPredictor()
        self.rollout = CounterfactualRolloutEvaluator(self.config.rollout)
        self.diagnostics = TutorEpisodeDiagnostics()
        self.last_waypoint_step = -10**9
        self._last_profile_update_step: int | None = None

    def reset(self) -> None:
        self.profile_belief = uniform_profile_belief(self.profiles)
        self.diagnostics = TutorEpisodeDiagnostics()
        self.last_waypoint_step = -10**9
        self._last_profile_update_step = None

    # ------------------------------------------------------------------
    # Posterior update
    # ------------------------------------------------------------------
    def _extract_observed_action(self, context: TutorDecisionContext) -> Any:
        snap = context.learner_policy_snapshot
        for name in ("action", "next_action", "chosen_action", "planned_action", "first_action"):
            value = get_any(snap, [name], None)
            if value is not None:
                return value
        actions = get_any(snap, ["actions", "planned_actions"], None)
        try:
            if actions:
                return actions[0]
        except Exception:
            pass
        hist = context.history
        for container_name in ("steps", "records", "events"):
            steps = get_any(hist, [container_name], None)
            try:
                if steps:
                    last = steps[-1]
                    for name in ("learner_action", "action", "chosen_action"):
                        value = get_any(last, [name], None)
                        if value is not None:
                            return value
            except Exception:
                pass
        return None

    def _profile_likelihoods(self, observed_action: Any, context: TutorDecisionContext) -> dict[str, float]:
        out: dict[str, float] = {}
        for profile in self.profiles:
            shadow = clone_from_snapshots(
                context.learner_memory_snapshot,
                context.learner_risk_belief_snapshot,
                profile,
                env_state=context.true_env_state,
                layout=context.true_layout,
            )
            paths = self.path_predictor.predict_topk(
                shadow,
                context.true_env_state,
                layout=context.true_layout,
                k=self.config.top_k_paths,
                horizon=min(5, self.config.rollout_horizon),
            )
            p = first_action_probability(paths, observed_action)
            out[profile.name] = likelihood_from_action_prob(p)
        return out

    def _maybe_update_profile_belief(self, context: TutorDecisionContext) -> None:
        try:
            step = step_count(context.true_env_state)
        except Exception:
            step = None
        if step is not None and self._last_profile_update_step == step:
            return
        observed_action = self._extract_observed_action(context)
        if observed_action is None:
            return
        likes = self._profile_likelihoods(observed_action, context)
        if not likes:
            return
        updated = {}
        alpha = max(0.0, min(1.0, self.config.posterior_update_strength))
        for name, prior in self.profile_belief.items():
            # Tempered Bayesian update; alpha < 1 prevents early lock-in.
            updated[name] = prior * (likes.get(name, 1e-4) ** alpha)
        self.profile_belief = normalize_belief(updated, floor=self.config.profile_belief_floor)
        self._last_profile_update_step = step

    # ------------------------------------------------------------------
    # Shadow particles and predictions
    # ------------------------------------------------------------------
    def _shadow_particles(self, context: TutorDecisionContext) -> list[tuple[ShadowLearnerState, float]]:
        particles: list[tuple[ShadowLearnerState, float]] = []
        for profile in self.profiles:
            w = self.profile_belief.get(profile.name, 0.0)
            if w <= 0.0:
                continue
            particles.append(
                (
                    clone_from_snapshots(
                        context.learner_memory_snapshot,
                        context.learner_risk_belief_snapshot,
                        profile,
                        env_state=context.true_env_state,
                        layout=context.true_layout,
                    ),
                    w,
                )
            )
        return particles

    def _mixture_predicted_paths(self, context: TutorDecisionContext, particles: list[tuple[ShadowLearnerState, float]]) -> list[PredictedPath]:
        paths: list[PredictedPath] = []
        for shadow, weight in particles:
            sub = self.path_predictor.predict_topk(
                shadow,
                context.true_env_state,
                layout=context.true_layout,
                k=self.config.top_k_paths,
                horizon=self.config.rollout_horizon,
            )
            for p in sub:
                q = PredictedPath(
                    cells=list(p.cells),
                    actions=list(p.actions),
                    probability=p.probability * weight,
                    predicted_cost=p.predicted_cost,
                    predicted_risk=p.predicted_risk,
                    predicted_info_gain=p.predicted_info_gain,
                    predicted_revisit_count=p.predicted_revisit_count,
                )
                paths.append(q)
        # Merge duplicate prefixes, adding probabilities and retaining lowest-cost diagnostics.
        by_key: dict[tuple[tuple[int, int], ...], PredictedPath] = {}
        for p in paths:
            key = tuple(p.cells[: min(4, len(p.cells))])
            if not key:
                continue
            if key not in by_key:
                by_key[key] = p
            else:
                old = by_key[key]
                old.probability += p.probability
                if p.predicted_cost < old.predicted_cost:
                    old.predicted_cost = p.predicted_cost
                    old.predicted_risk = p.predicted_risk
                    old.predicted_info_gain = p.predicted_info_gain
                    old.predicted_revisit_count = p.predicted_revisit_count
                    old.actions = p.actions
                    old.cells = p.cells
        merged = sorted(by_key.values(), key=lambda p: (-p.probability, p.predicted_cost))
        z = sum(p.probability for p in merged) or 1.0
        for p in merged:
            p.probability /= z
        return merged[: self.config.top_k_paths]

    # ------------------------------------------------------------------
    # Action selection and logs
    # ------------------------------------------------------------------
    def _score_candidates(self, candidates: list[Any], context: TutorDecisionContext, particles: list[tuple[ShadowLearnerState, float]]) -> list[tuple[Any, TutorActionValue]]:
        scored: list[tuple[Any, TutorActionValue]] = []
        for action in candidates:
            val = self.rollout.evaluate_candidate(action, context, particles)
            scored.append((action, val))
        return scored

    def _best_by_kind(self, scored: list[tuple[Any, TutorActionValue]], kind: str) -> tuple[Any | None, TutorActionValue | None]:
        kind = kind.upper()
        subset = [(a, v) for a, v in scored if str(get_any(a, ["kind"], "WAIT")).upper() == kind]
        if not subset:
            return None, None
        return max(subset, key=lambda av: av[1].q_total)

    def _select_with_guardrails(self, scored: list[tuple[Any, TutorActionValue]], step: int) -> tuple[Any, TutorActionValue]:
        wait_action, wait_val = self._best_by_kind(scored, "WAIT")
        if wait_action is None or wait_val is None:
            wait_action = make_tutor_action("WAIT", reason="fallback_no_wait_candidate")
            wait_val = TutorActionValue(q_total=0.0)
        # Enforce waypoint bandwidth before argmax.
        filtered: list[tuple[Any, TutorActionValue]] = []
        for a, v in scored:
            kind = str(get_any(a, ["kind"], "WAIT")).upper()
            if kind == "WAYPOINT" and step - self.last_waypoint_step < self.config.waypoint_cooldown_steps:
                continue
            filtered.append((a, v))
        if not filtered:
            filtered = [(wait_action, wait_val)]
        best_action, best_val = max(filtered, key=lambda av: av[1].q_total)
        best_kind = str(get_any(best_action, ["kind"], "WAIT")).upper()
        if best_kind == "WAYPOINT" and best_val.q_total < wait_val.q_total + self.config.waypoint_min_advantage_over_wait:
            return wait_action, wait_val
        if best_kind == "WARNING" and best_val.q_total < wait_val.q_total + self.config.warning_min_advantage_over_wait:
            return wait_action, wait_val
        if best_kind == "WAYPOINT":
            self.last_waypoint_step = step
        return best_action, best_val

    def _decision_log(self, step: int, selected: Any, selected_value: TutorActionValue, scored: list[tuple[Any, TutorActionValue]]) -> TutorDecisionLog:
        _, wait_val = self._best_by_kind(scored, "WAIT")
        _, warn_val = self._best_by_kind(scored, "WARNING")
        _, waypoint_val = self._best_by_kind(scored, "WAYPOINT")
        selected_kind = str(get_any(selected, ["kind"], "WAIT")).upper()
        log = TutorDecisionLog(
            step=step,
            selected_action=selected_kind,
            candidate_count=len(scored),
            q_wait=wait_val.q_total if wait_val else 0.0,
            q_best_warning=warn_val.q_total if warn_val else float("-inf"),
            q_best_waypoint=waypoint_val.q_total if waypoint_val else float("-inf"),
            predicted_p_death_wait=wait_val.p_death if wait_val else 0.0,
            predicted_p_timeout_wait=wait_val.p_timeout if wait_val else 0.0,
            predicted_map_gain_wait=wait_val.expected_map_gain if wait_val else 0.0,
            predicted_risk_ig_warning=warn_val.expected_risk_info_gain if warn_val else 0.0,
            predicted_assist_leakage=selected_value.assist_leakage,
            diagnostics=dict(selected_value.diagnostics),
        )
        return log

    def act(self, *args: Any, **kwargs: Any) -> Any:
        context = coerce_context(*args, **kwargs)
        phase = str(context.phase).lower()
        if self.config.eval_disabled and phase == "eval":
            return make_tutor_action("WAIT", reason="eval_phase_disabled", diagnostics={"eval_disabled": 1.0})

        if context.remaining_time <= 0 and context.true_env_state is not None:
            context.remaining_time = remaining_time_from_state(context.true_env_state, self.config.rollout_horizon)

        self._maybe_update_profile_belief(context)
        particles = self._shadow_particles(context)
        if not particles:
            return make_tutor_action("WAIT", reason="no_shadow_particles")

        predicted_paths = self._mixture_predicted_paths(context, particles)
        allow_waypoint = self.config.mode == "full"
        candidates = generate_tutor_candidates(
            context,
            predicted_paths,
            allow_waypoint=allow_waypoint,
            max_candidates=self.config.max_candidates,
        )
        if self.config.mode == "warning_only":
            candidates = [a for a in candidates if str(get_any(a, ["kind"], "WAIT")).upper() in {"WAIT", "WARNING"}]
        if not candidates:
            candidates = [make_tutor_action("WAIT", reason="no_candidates")]

        scored = self._score_candidates(candidates, context, particles)
        try:
            step = step_count(context.true_env_state)
        except Exception:
            step = len(self.diagnostics.decisions)
        selected, selected_val = self._select_with_guardrails(scored, step)
        log = self._decision_log(step, selected, selected_val, scored)
        self.diagnostics.append(log)

        diag = dict(get_any(selected, ["diagnostics"], {}) or {})
        diag.update(selected_val.diagnostics)
        diag.update(
            {
                "candidate_count": float(len(candidates)),
                "q_wait": log.q_wait,
                "q_best_warning": log.q_best_warning,
                "q_best_waypoint": log.q_best_waypoint,
                "selected_q": selected_val.q_total,
                "profile_entropy": self._profile_entropy(),
            }
        )
        try:
            selected.diagnostics = diag
        except Exception:
            pass
        return selected

    def _profile_entropy(self) -> float:
        import math

        h = 0.0
        for p in self.profile_belief.values():
            if p > 0:
                h -= p * math.log(p)
        return h


class WarningOnlyInverseTutor(InversePlanningTutor):
    def __init__(self, config: TutorConfig | None = None, profiles: list[LearnerProfile] | None = None):
        cfg = config or TutorConfig(mode="warning_only")
        cfg.mode = "warning_only"
        super().__init__(cfg, profiles=profiles)


class FullInverseTutor(InversePlanningTutor):
    def __init__(self, config: TutorConfig | None = None, profiles: list[LearnerProfile] | None = None):
        cfg = config or TutorConfig(mode="full")
        cfg.mode = "full"
        super().__init__(cfg, profiles=profiles)
