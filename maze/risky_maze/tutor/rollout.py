from __future__ import annotations

from dataclasses import dataclass, field
import copy
from types import SimpleNamespace
from typing import Any, Iterable

from .compat import Coord, get_any, make_tutor_action, safe_float
from .path_predictor import LearnerPathPredictor, _danger_probability
from .shadow import ShadowLearnerState
from .world_model import (
    cells_in_radius,
    current_pos,
    has_gem,
    hp_left,
    is_walkable,
    known_walls,
    known_walkable,
    mark_memory_observed,
    objective_coord,
    observed_vector,
    remaining_time_from_state,
    true_damage,
    visited_count,
)


@dataclass
class TutorUtilityWeights:
    success: float = 4.0
    death: float = 30.0
    damage: float = 5.0
    timeout: float = 8.0
    map_gain: float = 0.35
    risk_ig: float = 1.5
    eval_gain: float = 0.5
    cost: float = 1.0
    assist: float = 2.0
    boredom: float = 1.0


@dataclass
class TutorActionValue:
    q_total: float
    p_success: float = 0.0
    p_timeout: float = 0.0
    p_death: float = 0.0
    expected_damage: float = 0.0
    expected_steps: float = 0.0
    expected_map_gain: float = 0.0
    expected_risk_info_gain: float = 0.0
    expected_eval_gain_proxy: float = 0.0
    intervention_cost: float = 0.0
    assist_leakage: float = 0.0
    boredom_cost: float = 0.0
    p_catastrophe: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class RolloutConfig:
    horizon: int = 10
    view_radius: int = 2
    wait_cost: float = 0.0
    warning_cost: float = 0.2
    waypoint_cost: float = 0.8
    warning_leakage_scale: float = 0.2
    waypoint_leakage: float = 1.0
    catastrophe_damage_threshold: float = 2.0
    utility_weights: TutorUtilityWeights = field(default_factory=TutorUtilityWeights)


class CounterfactualRolloutEvaluator:
    def __init__(self, config: RolloutConfig | None = None):
        self.config = config or RolloutConfig()
        self.predictor = LearnerPathPredictor()

    def intervention_cost(self, action: Any) -> float:
        kind = str(get_any(action, ["kind"], "WAIT")).upper()
        if kind == "WARNING":
            return self.config.warning_cost
        if kind == "WAYPOINT":
            return self.config.waypoint_cost
        return self.config.wait_cost

    def assist_leakage(self, action: Any) -> float:
        kind = str(get_any(action, ["kind"], "WAIT")).upper()
        if kind == "WARNING":
            n = max(1, len(tuple(get_any(action, ["cells"], ()) or ())))
            return self.config.warning_leakage_scale * (1.0 / n)
        if kind == "WAYPOINT":
            base = safe_float(get_any(action, ["diagnostics"], {}).get("base_assist_leakage", self.config.waypoint_leakage) if isinstance(get_any(action, ["diagnostics"], {}), dict) else self.config.waypoint_leakage, self.config.waypoint_leakage)
            return base
        return 0.0

    def _warning_info_gain(self, action: Any, shadow: ShadowLearnerState, layout: Any) -> float:
        cells = tuple(get_any(action, ["cells"], ()) or ())
        if not cells or shadow.risk_belief_hat is None:
            return 0.0
        features: list[Any] = []
        for c in cells:
            feat = observed_vector(shadow.memory_hat, layout, c, allow_oracle=False)
            if feat is not None:
                features.append(feat)
        if not features:
            return 0.0
        before = [_danger_probability(shadow.risk_belief_hat, feat, default=0.25) for feat in features]
        updater = getattr(shadow.risk_belief_hat, "warning_update", None)
        if callable(updater):
            try:
                updater(features)
            except Exception:
                try:
                    updater(tuple(features))
                except Exception:
                    return 0.0
        else:
            return 0.0
        after = [_danger_probability(shadow.risk_belief_hat, feat, default=0.25) for feat in features]
        return sum(abs(a - b) for a, b in zip(after, before))

    def _path_has_danger(self, layout: Any, path: list[Coord]) -> float:
        if not path:
            return 0.0
        return 1.0 if any(true_damage(layout, c) > 0.0 for c in path) else 0.0

    def _maybe_supervised_risk_update(self, shadow: ShadowLearnerState, layout: Any, coord: Coord, damage: float) -> float:
        feat = observed_vector(shadow.memory_hat, layout, coord, allow_oracle=True)
        if feat is None or shadow.risk_belief_hat is None:
            return 0.0
        before = _danger_probability(shadow.risk_belief_hat, feat, default=0.25)
        label = 1 if damage > 0.0 else 0
        for name in ("update_labeled", "update_supervised", "update"):
            method = getattr(shadow.risk_belief_hat, name, None)
            if callable(method):
                try:
                    method(feat, label)
                    break
                except Exception:
                    try:
                        method([feat], [label])
                        break
                    except Exception:
                        pass
        after = _danger_probability(shadow.risk_belief_hat, feat, default=0.25)
        return abs(after - before)

    def _simulate_particle(self, action: Any, context: Any, shadow_in: ShadowLearnerState, weight: float) -> TutorActionValue:
        shadow = shadow_in.clone()
        layout = context.true_layout
        state = context.true_env_state
        try:
            pos = current_pos(state)
        except Exception:
            return TutorActionValue(q_total=0.0)
        hp = hp_left(state, default=1.0)
        carries_gem = has_gem(state)
        remaining = int(get_any(context, ["remaining_time"], 0) or remaining_time_from_state(state, self.config.horizon))
        horizon = max(1, min(self.config.horizon, remaining if remaining > 0 else self.config.horizon))

        kind = str(get_any(action, ["kind"], "WAIT")).upper()
        risk_ig = 0.0
        actionability = 0.0
        action_changed = 0.0
        path_before_has_danger = 0.0
        path_after_has_danger = 0.0
        hp0 = max(0.0, hp)
        catastrophic = 0.0
        if kind == "WARNING":
            pred_before = self.predictor.predict_topk(
                shadow,
                state,
                layout=layout,
                k=1,
                horizon=max(1, horizon),
            )
            before_cells = list(pred_before[0].cells) if pred_before else []
            path_before_has_danger = self._path_has_danger(layout, before_cells)
            risk_ig += self._warning_info_gain(action, shadow, layout)
            pred_after = self.predictor.predict_topk(
                shadow,
                state,
                layout=layout,
                k=1,
                horizon=max(1, horizon),
            )
            after_cells = list(pred_after[0].cells) if pred_after else []
            path_after_has_danger = self._path_has_danger(layout, after_cells)
            if before_cells and after_cells and tuple(before_cells[:3]) != tuple(after_cells[:3]):
                action_changed = 1.0
            if path_before_has_danger > 0.0 and path_after_has_danger <= 0.0:
                actionability = 1.0
        waypoint = get_any(action, ["waypoint"], None) if kind == "WAYPOINT" else None
        if waypoint is not None:
            waypoint = tuple(waypoint)

        memory_walkable = getattr(shadow.memory_hat, "known_walkable", None)
        if isinstance(memory_walkable, set):
            seen_before = set(memory_walkable)
        else:
            seen_before = set(known_walkable(shadow.memory_hat))
        memory_walls = getattr(shadow.memory_hat, "known_walls", None)
        if isinstance(memory_walls, set):
            seen_before.update(memory_walls)
        else:
            seen_before.update(known_walls(shadow.memory_hat))
        damage_sum = 0.0
        map_gain = 0.0
        boredom = 0.0
        steps = 0.0
        death = 0.0
        timeout = 0.0
        success = 0.0
        prev_dist = None
        cached_target: Coord | None = None
        cached_path: list[Coord] = []
        replans = 0.0

        for t in range(horizon):
            if waypoint is not None and pos == waypoint:
                waypoint = None
            target = waypoint or objective_coord(layout, SimpleNamespace(pos=pos, has_gem=carries_gem))
            shadow_target = getattr(shadow.objective_state_hat, "target", None)
            if shadow_target is not None:
                target = waypoint or shadow_target
            if target is None:
                boredom += 1.0
                break
            dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            if cached_target != target or not cached_path:
                path_state = SimpleNamespace(pos=pos, has_gem=carries_gem, layout=layout)
                shadow.objective_state_hat.target = target
                pred = self.predictor.predict_topk(
                    shadow,
                    path_state,
                    layout=layout,
                    k=1,
                    horizon=max(1, horizon - t),
                    waypoint=target,
                )
                replans += 1.0
                if not pred or not pred[0].cells:
                    boredom += 1.0
                    break
                cached_path = list(pred[0].cells)
                cached_target = target
            nxt = cached_path.pop(0)
            if not is_walkable(layout, nxt):
                boredom += 1.0
                cached_path = []
                break

            pos = nxt
            steps += 1.0
            dmg = true_damage(layout, pos)
            damage_sum += dmg
            hp -= dmg
            risk_ig += self._maybe_supervised_risk_update(shadow, layout, pos, dmg)
            if dmg > 0.0:
                cached_path = []
            if hp0 > 0.0 and (hp <= 0.0 or damage_sum >= max(hp0, self.config.catastrophe_damage_threshold)):
                catastrophic = 1.0

            newly_seen_this_step = 0
            for c in cells_in_radius(layout, pos, self.config.view_radius):
                if c in seen_before:
                    continue
                if is_walkable(layout, c):
                    newly_seen_this_step += 1
                seen_before.add(c)
                mark_memory_observed(shadow.memory_hat, layout, c, allow_oracle_feature=True)
            map_gain += newly_seen_this_step

            vc = getattr(shadow.memory_hat, "visited_count", None)
            if isinstance(vc, dict):
                vc[pos] = int(vc.get(pos, 0) or 0) + 1

            completed_all = False
            objective_hat = getattr(shadow, "objective_state_hat", None)
            if objective_hat is not None and getattr(objective_hat, "sequence", None):
                advance = getattr(objective_hat, "advance_if_reached", None)
                if callable(advance):
                    try:
                        completed_all = bool(advance(pos))
                    except Exception:
                        completed_all = False
                carries_gem = bool(getattr(objective_hat, "has_gem", carries_gem))
                if completed_all:
                    success = 1.0
                    break
            else:
                gem_coord = getattr(layout, "gem", getattr(layout, "gem_coord", None))
                exit_coord = getattr(layout, "exit", getattr(layout, "exit_coord", None))
                if gem_coord is not None and tuple(gem_coord) == pos:
                    carries_gem = True
                if carries_gem and exit_coord is not None and tuple(exit_coord) == pos:
                    success = 1.0
                    break
            if hp <= 0.0:
                death = 1.0
                catastrophic = 1.0
                break

            new_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            no_progress = prev_dist is not None and new_dist >= prev_dist
            repeated = visited_count(shadow.memory_hat, pos) > 1
            if newly_seen_this_step == 0 and no_progress and repeated:
                boredom += 1.0
            prev_dist = new_dist

        if success <= 0.0 and death <= 0.0 and remaining > 0 and steps >= remaining:
            timeout = 1.0
        eval_gain = map_gain + risk_ig - damage_sum - self.assist_leakage(action)
        return TutorActionValue(
            q_total=0.0,
            p_success=success * weight,
            p_timeout=timeout * weight,
            p_death=death * weight,
            expected_damage=damage_sum * weight,
            expected_steps=steps * weight,
            expected_map_gain=map_gain * weight,
            expected_risk_info_gain=risk_ig * weight,
            expected_eval_gain_proxy=eval_gain * weight,
            intervention_cost=self.intervention_cost(action) * weight,
            assist_leakage=self.assist_leakage(action) * weight,
            boredom_cost=boredom * weight,
            p_catastrophe=catastrophic * weight,
            diagnostics={
                "particle_weight": weight,
                "replans": replans,
                "warning_actionability": actionability,
                "warning_path_changed": action_changed,
                "warning_path_before_has_danger": path_before_has_danger,
                "warning_path_after_has_danger": path_after_has_danger,
            },
        )

    def evaluate_candidate(
        self,
        action: Any,
        context: Any,
        shadow_particles: Iterable[tuple[ShadowLearnerState, float]] | Iterable[ShadowLearnerState],
    ) -> TutorActionValue:
        totals = TutorActionValue(q_total=0.0)
        total_weight = 0.0
        warning_actionability = 0.0
        warning_path_changed = 0.0
        warning_path_before_has_danger = 0.0
        warning_path_after_has_danger = 0.0
        replans = 0.0
        for item in shadow_particles:
            if isinstance(item, tuple):
                shadow, weight = item
            else:
                shadow, weight = item, 1.0
            weight = max(0.0, float(weight))
            if weight <= 0.0:
                continue
            val = self._simulate_particle(action, context, shadow, weight)
            total_weight += weight
            totals.p_success += val.p_success
            totals.p_timeout += val.p_timeout
            totals.p_death += val.p_death
            totals.expected_damage += val.expected_damage
            totals.expected_steps += val.expected_steps
            totals.expected_map_gain += val.expected_map_gain
            totals.expected_risk_info_gain += val.expected_risk_info_gain
            totals.expected_eval_gain_proxy += val.expected_eval_gain_proxy
            totals.intervention_cost += val.intervention_cost
            totals.assist_leakage += val.assist_leakage
            totals.boredom_cost += val.boredom_cost
            totals.p_catastrophe += val.p_catastrophe
            warning_actionability += float(val.diagnostics.get("warning_actionability", 0.0)) * weight
            warning_path_changed += float(val.diagnostics.get("warning_path_changed", 0.0)) * weight
            warning_path_before_has_danger += float(val.diagnostics.get("warning_path_before_has_danger", 0.0)) * weight
            warning_path_after_has_danger += float(val.diagnostics.get("warning_path_after_has_danger", 0.0)) * weight
            replans += float(val.diagnostics.get("replans", 0.0)) * weight
        if total_weight <= 0.0:
            total_weight = 1.0
        # Values above are already weighted by probability.  Normalize in case
        # caller provided non-normalized weights.
        for name in (
            "p_success",
            "p_timeout",
            "p_death",
            "expected_damage",
            "expected_steps",
            "expected_map_gain",
            "expected_risk_info_gain",
            "expected_eval_gain_proxy",
            "intervention_cost",
            "assist_leakage",
            "boredom_cost",
            "p_catastrophe",
        ):
            setattr(totals, name, getattr(totals, name) / total_weight)

        w = self.config.utility_weights
        totals.q_total = (
            w.success * totals.p_success
            - w.death * totals.p_death
            - w.damage * totals.expected_damage
            - w.timeout * totals.p_timeout
            + w.map_gain * totals.expected_map_gain
            + w.risk_ig * totals.expected_risk_info_gain
            + w.eval_gain * totals.expected_eval_gain_proxy
            - w.cost * totals.intervention_cost
            - w.assist * totals.assist_leakage
            - w.boredom * totals.boredom_cost
        )
        totals.diagnostics = {
            "q_total": totals.q_total,
            "q_learning_value": w.map_gain * totals.expected_map_gain
            + w.risk_ig * totals.expected_risk_info_gain
            + w.eval_gain * totals.expected_eval_gain_proxy,
            "q_teach_cost": w.damage * totals.expected_damage + w.timeout * totals.p_timeout,
            "q_assist_cost": w.cost * totals.intervention_cost + w.assist * totals.assist_leakage,
            "q_boredom_cost": w.boredom * totals.boredom_cost,
            "p_success": totals.p_success,
            "p_timeout": totals.p_timeout,
            "p_death": totals.p_death,
            "p_catastrophe": totals.p_catastrophe,
            "expected_damage": totals.expected_damage,
            "expected_steps": totals.expected_steps,
            "expected_map_gain": totals.expected_map_gain,
            "expected_risk_info_gain": totals.expected_risk_info_gain,
            "expected_eval_gain_proxy": totals.expected_eval_gain_proxy,
            "intervention_cost": totals.intervention_cost,
            "assist_leakage": totals.assist_leakage,
            "boredom_cost": totals.boredom_cost,
            "warning_actionability": warning_actionability / total_weight,
            "warning_path_changed": warning_path_changed / total_weight,
            "warning_path_before_has_danger": warning_path_before_has_danger / total_weight,
            "warning_path_after_has_danger": warning_path_after_has_danger / total_weight,
            "replans": replans / total_weight,
        }
        return totals
