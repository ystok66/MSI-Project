from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .candidates import generate_tutor_candidates, generate_waypoint_candidates
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
    mode: str = "full"
    top_k_paths: int = 4
    rollout_horizon: int = 10
    max_candidates: int = 20
    profile_belief_floor: float = 1e-4
    eval_disabled: bool = True
    waypoint_cooldown_steps: int = 4
    max_waypoints_per_episode: int = 3
    frontier_only_waypoint: bool = False
    waypoint_min_advantage_over_wait: float = 0.10
    warning_min_advantage_over_wait: float = 0.00
    warning_actionability_threshold: float = 0.0
    waypoint_damage_veto_margin: float = float("inf")
    posterior_update_strength: float = 1.0
    safety_shield_enabled: bool = False
    catastrophe_threshold: float = 0.35
    scaffold_waypoint_types: tuple[str, ...] = ("frontier", "landmark", "bottleneck")
    randomize_scaffold_choice: bool = False
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
        self.waypoints_used = 0
        self._last_profile_update_step: int | None = None

    def reset(self) -> None:
        self.profile_belief = uniform_profile_belief(self.profiles)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.diagnostics = TutorEpisodeDiagnostics()
        self.last_waypoint_step = -10**9
        self.waypoints_used = 0
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

    def _build_profile_path_cache(self, context: TutorDecisionContext) -> list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]]:
        cache: list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]] = []
        if not self.profiles:
            return cache
        base_shadow = clone_from_snapshots(
            context.learner_memory_snapshot,
            context.learner_risk_belief_snapshot,
            self.profiles[0],
            env_state=context.true_env_state,
            layout=context.true_layout,
        )
        shared_memory = base_shadow.memory_hat
        shared_belief = base_shadow.risk_belief_hat
        base_objective_state = base_shadow.objective_state_hat
        for idx, profile in enumerate(self.profiles):
            if idx == 0:
                shadow = base_shadow
            else:
                shadow = ShadowLearnerState(
                    memory_hat=shared_memory,
                    risk_belief_hat=shared_belief,
                    objective_state_hat=copy.deepcopy(base_objective_state),
                    profile=profile,
                )
            paths = self.path_predictor.predict_topk(
                shadow,
                context.true_env_state,
                layout=context.true_layout,
                k=self.config.top_k_paths,
                horizon=self.config.rollout_horizon,
            )
            cache.append((profile, shadow, paths))
        return cache

    def _profile_likelihoods_from_cache(
        self,
        observed_action: Any,
        profile_cache: list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for profile, _shadow, paths in profile_cache:
            p = first_action_probability(paths, observed_action)
            out[profile.name] = likelihood_from_action_prob(p)
        return out

    def _maybe_update_profile_belief(
        self,
        context: TutorDecisionContext,
        profile_cache: list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]],
    ) -> None:
        try:
            step = step_count(context.true_env_state)
        except Exception:
            step = None
        if step is not None and self._last_profile_update_step == step:
            return
        observed_action = self._extract_observed_action(context)
        if observed_action is None:
            return
        likes = self._profile_likelihoods_from_cache(observed_action, profile_cache)
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
    def _shadow_particles_from_cache(
        self,
        profile_cache: list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]],
    ) -> list[tuple[ShadowLearnerState, float]]:
        particles: list[tuple[ShadowLearnerState, float]] = []
        for profile, shadow, _paths in profile_cache:
            w = self.profile_belief.get(profile.name, 0.0)
            if w <= 0.0:
                continue
            particles.append((shadow, w))
        return particles

    def _mixture_predicted_paths_from_cache(
        self,
        profile_cache: list[tuple[LearnerProfile, ShadowLearnerState, list[PredictedPath]]],
    ) -> list[PredictedPath]:
        paths: list[PredictedPath] = []
        for profile, _shadow, sub in profile_cache:
            weight = self.profile_belief.get(profile.name, 0.0)
            if weight <= 0.0:
                continue
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
    def _mode_key(self) -> str:
        return str(self.config.mode or "full").lower()

    def _legacy_competition_mode(self) -> bool:
        return self._mode_key() in {"warning_only", "full"}

    def _warning_candidates(self, context: TutorDecisionContext, predicted_paths: list[PredictedPath]) -> list[Any]:
        candidates = generate_tutor_candidates(
            context,
            predicted_paths,
            allow_waypoint=False,
            max_candidates=self.config.max_candidates,
        )
        return [a for a in candidates if str(get_any(a, ["kind"], "WAIT")).upper() in {"WAIT", "WARNING"}]

    def _scaffold_candidates(self, context: TutorDecisionContext, predicted_paths: list[PredictedPath]) -> list[Any]:
        candidates: list[Any] = [make_tutor_action("WAIT", reason="safe_wait_default")]
        candidates.extend(
            generate_waypoint_candidates(
                context,
                predicted_paths,
                max_candidates=max(1, self.config.max_candidates - 1),
                frontier_only=self.config.frontier_only_waypoint,
                allowed_types=tuple(self.config.scaffold_waypoint_types or ()),
            )
        )
        return candidates

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

    def _select_safety_action(self, scored: list[tuple[Any, TutorActionValue]]) -> tuple[Any | None, TutorActionValue | None, dict[str, float]]:
        wait_action, wait_val = self._best_by_kind(scored, "WAIT")
        if wait_action is None or wait_val is None:
            wait_action = make_tutor_action("WAIT", reason="fallback_no_wait_candidate")
            wait_val = TutorActionValue(q_total=0.0)
        diag: dict[str, float] = {
            "decision_layer_safety": 1.0,
            "safety_shield_triggered": 0.0,
            "wait_catastrophe": float(wait_val.p_catastrophe),
            "safe_alternative_exists": 0.0,
            "safe_candidate_count": 0.0,
            "safety_warning_unresolved": 0.0,
        }
        if wait_val.p_catastrophe <= self.config.catastrophe_threshold:
            return None, None, diag

        warnings = [(a, v) for a, v in scored if str(get_any(a, ["kind"], "WAIT")).upper() == "WARNING"]
        safe_warnings = [
            (a, v)
            for a, v in warnings
            if v.p_catastrophe <= self.config.catastrophe_threshold
        ]
        diag["safe_candidate_count"] = float(len(safe_warnings))
        diag["safe_alternative_exists"] = 1.0 if safe_warnings else 0.0

        def safety_key(item: tuple[Any, TutorActionValue]) -> tuple[float, float, float]:
            _action, value = item
            catastrophe_gain = float(wait_val.p_catastrophe) - float(value.p_catastrophe)
            actionability = float(value.diagnostics.get("warning_actionability", 0.0) or 0.0)
            return catastrophe_gain, actionability, float(value.q_total)

        if safe_warnings:
            best_action, best_val = max(safe_warnings, key=safety_key)
            diag["safety_shield_triggered"] = 1.0
            return best_action, best_val, diag
        if warnings:
            best_action, best_val = max(warnings, key=safety_key)
            diag["safety_shield_triggered"] = 1.0
            diag["safety_warning_unresolved"] = 1.0
            return best_action, best_val, diag
        diag["safety_warning_unresolved"] = 1.0
        return wait_action, wait_val, diag

    def _select_scaffold_action(self, scored: list[tuple[Any, TutorActionValue]], step: int) -> tuple[Any, TutorActionValue, dict[str, float]]:
        wait_action, wait_val = self._best_by_kind(scored, "WAIT")
        if wait_action is None or wait_val is None:
            wait_action = make_tutor_action("WAIT", reason="fallback_no_wait_candidate")
            wait_val = TutorActionValue(q_total=0.0)
        diag: dict[str, float] = {
            "decision_layer_scaffold": 1.0,
            "waypoint_budget_exhausted": 0.0,
            "waypoint_damage_veto_blocked": 0.0,
            "scaffold_candidate_count": 0.0,
            "scaffold_improving_waypoint_count": 0.0,
            "scaffold_wait_timeout_risk": float(wait_val.p_timeout),
            "scaffold_wait_boredom_cost": float(wait_val.boredom_cost),
            "scaffold_wait_success_prob": float(wait_val.p_success),
            "scaffold_trigger_no_progress_risk": 0.0,
            "scaffold_trigger_timeout_risk": 0.0,
        }
        if wait_val.p_timeout > 0.0:
            diag["scaffold_trigger_timeout_risk"] = 1.0
        if wait_val.boredom_cost > 0.0:
            diag["scaffold_trigger_no_progress_risk"] = 1.0
        improving: list[tuple[Any, TutorActionValue]] = []
        for action, value in scored:
            kind = str(get_any(action, ["kind"], "WAIT")).upper()
            if kind != "WAYPOINT":
                continue
            diag["scaffold_candidate_count"] += 1.0
            if step - self.last_waypoint_step < self.config.waypoint_cooldown_steps:
                continue
            if self.waypoints_used >= self.config.max_waypoints_per_episode:
                diag["waypoint_budget_exhausted"] = 1.0
                continue
            damage_margin = float(self.config.waypoint_damage_veto_margin)
            if value.expected_damage > wait_val.expected_damage + damage_margin:
                diag["waypoint_damage_veto_blocked"] += 1.0
                continue
            improves_timeout = value.p_timeout + 1e-9 < wait_val.p_timeout
            improves_boredom = value.boredom_cost + 1e-9 < wait_val.boredom_cost
            improves_success = value.p_success > wait_val.p_success
            improves_q = value.q_total >= wait_val.q_total + self.config.waypoint_min_advantage_over_wait
            if improves_timeout or improves_boredom or improves_success or improves_q:
                improving.append((action, value))
        diag["scaffold_improving_waypoint_count"] = float(len(improving))
        if not improving:
            return wait_action, wait_val, diag

        def scaffold_key(item: tuple[Any, TutorActionValue]) -> tuple[float, float, float, float]:
            _action, value = item
            return (
                float(value.assist_leakage),
                float(value.p_timeout),
                float(value.boredom_cost),
                -float(value.q_total),
            )

        if self.config.randomize_scaffold_choice:
            diag["randomized_scaffold_choice"] = 1.0
            frontier_items = [
                (action, value)
                for action, value in improving
                if str(get_any(action, ["diagnostics", "waypoint_type"], "")).lower() == "frontier"
            ]
            pool = frontier_items or improving
            best_action, best_val = pool[step % len(pool)]
        else:
            best_action, best_val = min(improving, key=scaffold_key)
        diag["waypoint_type"] = str(get_any(best_action, ["diagnostics", "waypoint_type"], ""))
        self.last_waypoint_step = step
        self.waypoints_used += 1
        return best_action, best_val, diag

    def _select_with_guardrails(self, scored: list[tuple[Any, TutorActionValue]], step: int) -> tuple[Any, TutorActionValue, dict[str, float]]:
        wait_action, wait_val = self._best_by_kind(scored, "WAIT")
        if wait_action is None or wait_val is None:
            wait_action = make_tutor_action("WAIT", reason="fallback_no_wait_candidate")
            wait_val = TutorActionValue(q_total=0.0)
        guardrail_diag: dict[str, float] = {
            "safety_shield_triggered": 0.0,
            "wait_catastrophe": float(wait_val.p_catastrophe),
            "waypoint_budget_exhausted": 0.0,
            "safe_alternative_exists": 0.0,
            "safe_candidate_count": 0.0,
            "warning_actionability_blocked": 0.0,
            "waypoint_damage_veto_blocked": 0.0,
        }
        # Enforce waypoint bandwidth before argmax.
        filtered: list[tuple[Any, TutorActionValue]] = []
        for a, v in scored:
            kind = str(get_any(a, ["kind"], "WAIT")).upper()
            if kind == "WAYPOINT" and step - self.last_waypoint_step < self.config.waypoint_cooldown_steps:
                continue
            if kind == "WAYPOINT" and self.waypoints_used >= self.config.max_waypoints_per_episode:
                guardrail_diag["waypoint_budget_exhausted"] = 1.0
                continue
            if kind == "WARNING":
                actionability = float(v.diagnostics.get("warning_actionability", 0.0) or 0.0)
                if actionability < float(self.config.warning_actionability_threshold):
                    guardrail_diag["warning_actionability_blocked"] += 1.0
                    continue
            if kind == "WAYPOINT":
                damage_margin = float(self.config.waypoint_damage_veto_margin)
                if v.expected_damage > wait_val.expected_damage + damage_margin:
                    guardrail_diag["waypoint_damage_veto_blocked"] += 1.0
                    continue
            filtered.append((a, v))
        if not filtered:
            filtered = [(wait_action, wait_val)]
        if self.config.safety_shield_enabled and wait_val.p_catastrophe > self.config.catastrophe_threshold:
            safe_candidates = [
                (a, v)
                for a, v in filtered
                if v.p_catastrophe <= self.config.catastrophe_threshold
                and str(get_any(a, ["kind"], "WAIT")).upper() != "WAIT"
            ]
            guardrail_diag["safe_candidate_count"] = float(len(safe_candidates))
            guardrail_diag["safe_alternative_exists"] = 1.0 if safe_candidates else 0.0
            if safe_candidates:
                best_action, best_val = max(safe_candidates, key=lambda av: av[1].q_total)
                if str(get_any(best_action, ["kind"], "WAIT")).upper() == "WAYPOINT":
                    self.last_waypoint_step = step
                    self.waypoints_used += 1
                guardrail_diag["safety_shield_triggered"] = 1.0
                return best_action, best_val, guardrail_diag
        best_action, best_val = max(filtered, key=lambda av: av[1].q_total)
        best_kind = str(get_any(best_action, ["kind"], "WAIT")).upper()
        if best_kind == "WAYPOINT" and best_val.q_total < wait_val.q_total + self.config.waypoint_min_advantage_over_wait:
            return wait_action, wait_val, guardrail_diag
        if best_kind == "WARNING" and best_val.q_total < wait_val.q_total + self.config.warning_min_advantage_over_wait:
            return wait_action, wait_val, guardrail_diag
        if best_kind == "WAYPOINT":
            self.last_waypoint_step = step
            self.waypoints_used += 1
        return best_action, best_val, guardrail_diag

    def _finalize_action(
        self,
        *,
        step: int,
        selected: Any,
        selected_val: TutorActionValue,
        scored: list[tuple[Any, TutorActionValue]],
        extra_diag: dict[str, Any] | None = None,
    ) -> Any:
        log = self._decision_log(step, selected, selected_val, scored)
        self.diagnostics.append(log)
        diag = dict(get_any(selected, ["diagnostics"], {}) or {})
        diag.update(selected_val.diagnostics)
        diag.update(extra_diag or {})
        diag.update(
            {
                "candidate_count": float(len(scored)),
                "q_wait": log.q_wait,
                "q_best_warning": log.q_best_warning,
                "q_best_waypoint": log.q_best_waypoint,
                "selected_q": selected_val.q_total,
                "profile_entropy": self._profile_entropy(),
                "selected_p_catastrophe": selected_val.p_catastrophe,
                "waypoints_used": float(self.waypoints_used),
            }
        )
        return make_tutor_action(
            str(get_any(selected, ["kind"], "WAIT")),
            cells=tuple(get_any(selected, ["cells"], ()) or ()),
            waypoint=get_any(selected, ["waypoint"], None),
            reason=str(get_any(selected, ["reason"], "")),
            diagnostics=diag,
        )

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

        profile_cache = self._build_profile_path_cache(context)
        self._maybe_update_profile_belief(context, profile_cache)
        particles = self._shadow_particles_from_cache(profile_cache)
        if not particles:
            return make_tutor_action("WAIT", reason="no_shadow_particles")

        predicted_paths = self._mixture_predicted_paths_from_cache(profile_cache)
        try:
            step = step_count(context.true_env_state)
        except Exception:
            step = len(self.diagnostics.decisions)
        if self._legacy_competition_mode():
            allow_waypoint = self._mode_key() == "full"
            candidates = generate_tutor_candidates(
                context,
                predicted_paths,
                allow_waypoint=allow_waypoint,
                max_candidates=self.config.max_candidates,
                frontier_only_waypoint=self.config.frontier_only_waypoint,
            )
            if self._mode_key() == "warning_only":
                candidates = [a for a in candidates if str(get_any(a, ["kind"], "WAIT")).upper() in {"WAIT", "WARNING"}]
            if not candidates:
                candidates = [make_tutor_action("WAIT", reason="no_candidates")]
            scored = self._score_candidates(candidates, context, particles)
            selected, selected_val, guardrail_diag = self._select_with_guardrails(scored, step)
            return self._finalize_action(
                step=step,
                selected=selected,
                selected_val=selected_val,
                scored=scored,
                extra_diag=guardrail_diag,
            )

        safety_candidates = self._warning_candidates(context, predicted_paths)
        safety_scored = self._score_candidates(safety_candidates, context, particles)
        safety_action, safety_val, safety_diag = self._select_safety_action(safety_scored)
        if safety_action is not None and safety_val is not None:
            return self._finalize_action(
                step=step,
                selected=safety_action,
                selected_val=safety_val,
                scored=safety_scored,
                extra_diag=safety_diag,
            )

        if self._mode_key() == "safety_shield_only":
            wait_action, wait_val = self._best_by_kind(safety_scored, "WAIT")
            if wait_action is None or wait_val is None:
                wait_action = make_tutor_action("WAIT", reason="safe_wait_default")
                wait_val = TutorActionValue(q_total=0.0)
            return self._finalize_action(
                step=step,
                selected=wait_action,
                selected_val=wait_val,
                scored=safety_scored,
                extra_diag=safety_diag,
            )

        scaffold_candidates = self._scaffold_candidates(context, predicted_paths)
        scaffold_scored = self._score_candidates(scaffold_candidates, context, particles)
        selected, selected_val, scaffold_diag = self._select_scaffold_action(scaffold_scored, step)
        scaffold_diag.update(
            {
                "decision_layer_safety": 0.0,
                "decision_layer_scaffold": 1.0,
            }
        )
        return self._finalize_action(
            step=step,
            selected=selected,
            selected_val=selected_val,
            scored=scaffold_scored,
            extra_diag=scaffold_diag,
        )

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


class SafetyShieldOnlyTutor(InversePlanningTutor):
    def __init__(self, config: TutorConfig | None = None, profiles: list[LearnerProfile] | None = None):
        cfg = config or TutorConfig(mode="safety_shield_only")
        cfg.mode = "safety_shield_only"
        cfg.safety_shield_enabled = True
        super().__init__(cfg, profiles=profiles)


class SafetyScaffoldTutor(InversePlanningTutor):
    def __init__(self, config: TutorConfig | None = None, profiles: list[LearnerProfile] | None = None):
        cfg = config or TutorConfig(mode="shield_plus_minimal_waypoint")
        cfg.mode = "shield_plus_minimal_waypoint"
        cfg.safety_shield_enabled = True
        super().__init__(cfg, profiles=profiles)
