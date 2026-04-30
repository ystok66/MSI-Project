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
    feature_at,
    has_gem,
    hp_left,
    is_walkable,
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
        if kind == "WARNING":
            risk_ig += self._warning_info_gain(action, shadow, layout)
        waypoint = get_any(action, ["waypoint"], None) if kind == "WAYPOINT" else None
        if waypoint is not None:
            waypoint = tuple(waypoint)

        seen_before = set(known_walkable(shadow.memory_hat))
        damage_sum = 0.0
        map_gain = 0.0
        boredom = 0.0
        steps = 0.0
        death = 0.0
        timeout = 0.0
        success = 0.0
        prev_dist = None

        for t in range(horizon):
            target = waypoint or objective_coord(layout, SimpleNamespace(pos=pos, has_gem=carries_gem))
            if target is None:
                boredom += 1.0
                break
            dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            path_state = SimpleNamespace(pos=pos, has_gem=carries_gem, layout=layout)
            shadow.objective_state_hat.target = target
            pred = self.predictor.predict_topk(shadow, path_state, layout=layout, k=1, horizon=max(1, horizon - t), waypoint=target)
            if not pred or not pred[0].cells:
                boredom += 1.0
                break
            nxt = pred[0].cells[0]
            if not is_walkable(layout, nxt):
                boredom += 1.0
                break

            pos = nxt
            steps += 1.0
            dmg = true_damage(layout, pos)
            damage_sum += dmg
            hp -= dmg
            risk_ig += self._maybe_supervised_risk_update(shadow, layout, pos, dmg)

            newly_seen_this_step = 0
            for c in cells_in_radius(layout, pos, self.config.view_radius):
                if c not in seen_before and is_walkable(layout, c):
                    newly_seen_this_step += 1
                    seen_before.add(c)
                mark_memory_observed(shadow.memory_hat, layout, c, allow_oracle_feature=True)
            map_gain += newly_seen_this_step

            vc = getattr(shadow.memory_hat, "visited_count", None)
            if isinstance(vc, dict):
                vc[pos] = int(vc.get(pos, 0) or 0) + 1

            gem_coord = getattr(layout, "gem", getattr(layout, "gem_coord", None))
            exit_coord = getattr(layout, "exit", getattr(layout, "exit_coord", None))
            if gem_coord is not None and tuple(gem_coord) == pos:
                carries_gem = True
            if carries_gem and exit_coord is not None and tuple(exit_coord) == pos:
                success = 1.0
                break
            if hp <= 0.0:
                death = 1.0
                break

            new_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            no_progress = prev_dist is not None and new_dist >= prev_dist
            repeated = visited_count(shadow.memory_hat, pos) > 1
            if newly_seen_this_step == 0 and no_progress and repeated:
                boredom += 1.0
            prev_dist = dist

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
            diagnostics={"particle_weight": weight},
        )

    def evaluate_candidate(
        self,
        action: Any,
        context: Any,
        shadow_particles: Iterable[tuple[ShadowLearnerState, float]] | Iterable[ShadowLearnerState],
    ) -> TutorActionValue:
        totals = TutorActionValue(q_total=0.0)
        total_weight = 0.0
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
            "p_success": totals.p_success,
            "p_timeout": totals.p_timeout,
            "p_death": totals.p_death,
            "expected_damage": totals.expected_damage,
            "expected_steps": totals.expected_steps,
            "expected_map_gain": totals.expected_map_gain,
            "expected_risk_info_gain": totals.expected_risk_info_gain,
            "expected_eval_gain_proxy": totals.expected_eval_gain_proxy,
            "intervention_cost": totals.intervention_cost,
            "assist_leakage": totals.assist_leakage,
            "boredom_cost": totals.boredom_cost,
        }
        return totals
