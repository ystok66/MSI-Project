"""
Lattice V2 Runner — episode orchestration for the V2 platform.

Architecture
============
This is the canonical episode runner for the pedagogical_ip project (Phase 2+).
It provides explicit reset/step semantics with no Gym dependency.

Step flow
---------
Each call to step(s) executes three sub-steps IN ORDER:
  1. observe(s)     — agent observes features at current position
  2. apply_tutor(s) — teacher may WAIT/WARN/UNLOCK/ITEM_DROP
  3. plan_and_move(s) — agent plans via A*, moves, outcome resolves

Mode switches (cumulative, additive)
-------------------------------------
- latent_mode=True          → Phase 4: LatentCostRiskHead for cost/risk prediction
- patch_radius>1            → Phase 5: extended observation radius
- prefix_horizon>0          → Phase 5: diagnostic prefix predictions
- belief_planning_mode=True → Phase 6: BeliefPlan with score breakdown
- robot_belief_mode=True    → Phase 7: robot belief + counterfactual intervention
- intervention_family_mode  → Phase 8: unified 4-action scoring + shield
- allowed_interventions     → Cleanup: experiment-condition enforcement

Canonical V2 path
-----------------
  runner = LatticeV2Runner()
  state = runner.reset(seed=42, latent_mode=True, ...)
  while not state.done:
      state = runner.step(state)
  metrics = runner.get_metrics(state)

Legacy baselines are preserved when all Phase 4+ switches are False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..envs.lattice_v2 import (
    generate_lattice_v2, LatticeV2Meta, SegmentMeta, FEATURE_DIM,
)
from ..envs.map_generator import CellType, GridMap
from ..envs.map_families import FamilyConfig
from ..agents.feature_belief import FeatureBeliefMap
from ..agents.risk_model import BayesianRiskHead
from ..agents.cost_risk_model import LatentCostRiskHead
from ..agents.observation_model import observe_features, observe_features_patch
from ..agents.planner_astar import plan_next_action_v2
from ..agents.prefix_prediction import compute_prefix_predictions, PrefixPrediction
from ..agents.belief_planning import (
    plan_from_belief, estimate_failure_modes,
    BeliefPlan, FailureModeEstimate,
)
from ..agents.warning_update import (
    Utterance, apply_warning, select_best_warning_action_gap,
    apply_warning_dispatch, VALID_WARNING_VARIANTS,
    map_segment_to_rsa_context, map_legacy_to_rsa_utterance,
    apply_planner_adapter, apply_pseudolabel_adapter,
)
from ..agents.rsa_warning_channel import (
    RSAWarningChannel, RSABeliefState, compute_warning_belief_delta,
)
from ..agents.stochastic_agent_policy import (
    BranchAttributes, compute_choice_probs, AgentPolicyParams,
)
from ..teachers.time_aware_door_tutor import TimeAwareDoorTutor
from ..teachers.robot_belief import (
    RobotBelief, init_robot_belief, sync_robot_belief,
)
from ..teachers.intervention_policy import (
    score_interventions, InterventionDecision, InterventionConfig,
)
from ..teachers.interventions import InventoryState, SHIELD_DEFAULT_RISK_REDUCTION
from ..teachers.perceptual_model import (
    PerceptualAccessState, init_perceptual_access, update_perceptual_access,
)


@dataclass
class V2EpisodeState:
    """All mutable state for one V2 episode.

    Field groups
    ------------
    Grid:      gridmap, config, meta — environment structure (frozen after reset)
    Agent:     agent_pos, goal — agent position and target
    Beliefs:   feature_belief (Kalman over 4D features), risk_head, belief_cost, passable
    Time:      t (current step), t_max (deadline)
    Teacher:   tutor, tutor_mode, warning_mode, warned_*, lambda_lane_warn, closure_budget
    Latent:    latent_mode, latent_predictor (LatentCostRiskHead)
    Phase 5:   patch_radius, prefix_horizon, last_prefix
    Phase 6:   belief_planning_mode, confidence_temperature, last_belief_plan, last_failure_modes
    Phase 7:   robot_belief_mode, robot_belief, last_intervention, belief_copy_mode, budget_mismatch
    Phase 8:   intervention_family_mode, item_drop_enabled, inventory, allowed_interventions
    Metrics:   survived, reached_goal, done, steps, unlock_count, warn_count,
               risky_entered, traps_hit, cue_cells_seen
    """

    # Grid
    gridmap: GridMap
    config: FamilyConfig
    meta: LatticeV2Meta

    # Agent
    agent_pos: tuple[int, int]
    goal: tuple[int, int]

    # Beliefs
    feature_belief: FeatureBeliefMap
    risk_head: BayesianRiskHead
    belief_cost: np.ndarray       # (H, W)
    passable: np.ndarray          # (H, W) bool

    # Time
    t: int                        # current step (0-indexed)
    t_max: int                    # max steps

    # Teacher state
    tutor: Optional[TimeAwareDoorTutor]
    tutor_mode: str               # "none", "time_aware", "warn_first", "always_close"
    warning_mode: str             # "none", "fixed", "selected"
    warned_segments: set = field(default_factory=set)
    warned_lane_bias: dict = field(default_factory=dict)
    warned_cell_extra: dict = field(default_factory=dict)
    lambda_lane_warn: float = 5.0
    closure_budget: Optional[int] = None

    # Latent mode
    latent_mode: bool = False
    latent_predictor: Optional[LatentCostRiskHead] = None

    # Patch / prefix config (Phase 5)
    patch_radius: int = 1           # 0=self only, 1=legacy, 2=extended
    prefix_horizon: int = 0         # 0=disabled
    last_prefix: Optional[PrefixPrediction] = None

    # Belief planning config (Phase 6)
    belief_planning_mode: bool = False
    confidence_temperature: float = 1.0
    last_belief_plan: Optional[BeliefPlan] = None
    last_failure_modes: Optional[FailureModeEstimate] = None

    # Robot belief config (Phase 7)
    robot_belief_mode: bool = False
    robot_belief: Optional[RobotBelief] = None
    last_intervention: Optional[InterventionDecision] = None
    belief_copy_mode: str = "exact"
    budget_mismatch: int = 0

    # Intervention family config (Phase 8)
    intervention_family_mode: bool = False
    item_drop_enabled: bool = False
    inventory: Optional[InventoryState] = None
    allowed_interventions: Optional[frozenset] = None  # None = all allowed
    gate_mode: str = "block_risky"  # "block_risky" | "unlock_shortcut"

    # Tutor Perceptual Model (Phase 10)
    perceptual_access: Optional[PerceptualAccessState] = None

    # Step 2: RSA Warning Channel
    warning_variant: str = "rsa_obs_s1"  # Phase 1A canonical; see VALID_WARNING_VARIANTS
    rsa_channel: Optional[object] = None           # RSAWarningChannel instance
    rsa_belief_state: Optional[object] = None      # RSABeliefState instance
    rsa_warn_diagnostics: list = field(default_factory=list)  # per-warning RSA info

    # GTET-L factor ablation (Phase GTET)
    factor_mode: str = "G_THETA"  # Phase 1: z demoted; see FACTOR_MODES in gtet_factor_adapter
    predictor_mode: str = "P4"  # "P1"=E[z], "P2"=MAP, "P3"=route_mix, "P4"=z_masked (temp default)
    gtet_posterior: Optional[object] = None  # JointGoalPrefPosterior (GTET only)
    gtet_action_log: list = field(default_factory=list)  # [(t, intervention_action), ...]

    # Phase 1B: boredom penalty weight (canonical: 0.3)
    boredom_weight: float = 0.3

    # Metrics
    survived: bool = True
    reached_goal: bool = False
    done: bool = False
    steps: int = 0
    unlock_count: int = 0
    warn_count: int = 0
    risky_entered: int = 0
    traps_hit: int = 0
    cue_cells_seen: int = 0

    # Phase 0 audit: read-only instrumentation (zero overhead when False)
    audit_mode: bool = False
    audit_trace: list = field(default_factory=list)

    # RNG
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


def _build_warned_cell_extra(
    warned_lane_bias: dict,
    meta: LatticeV2Meta,
) -> dict:
    """Convert segment-level bias to per-cell extra cost dict."""
    out = {}
    for seg_idx, bias_val in warned_lane_bias.items():
        seg = meta.segments[seg_idx]
        out[seg.risky_entry_gate] = bias_val
        for rc in seg.risky_cells:
            out[rc] = bias_val
    return out


class LatticeV2Runner:
    """Episode runner for the Lattice V2 platform.

    Provides explicit reset()/step() semantics. No Gym dependency.
    Teacher cadence: tutor acts every step but only triggers when
    agent is in row 2 near a segment entry.
    """

    def reset(
        self,
        seed: int,
        difficulty: str = "medium",
        tutor_mode: str = "none",
        time_ratio: float = 1.3,
        closure_budget: Optional[int] = None,
        warning_mode: str = "none",
        risk_head: Optional[BayesianRiskHead] = None,
        lambda_lane_warn: float = 5.0,
        latent_mode: bool = False,
        latent_predictor: Optional[LatentCostRiskHead] = None,
        risk_supervision: str = "oracle_visited",
        patch_radius: int = 1,
        prefix_horizon: int = 0,
        belief_planning_mode: bool = False,
        confidence_temperature: float = 1.0,
        robot_belief_mode: bool = False,
        belief_copy_mode: str = "exact",
        budget_mismatch: int = 0,
        intervention_family_mode: bool = False,
        item_drop_enabled: bool = False,
        shield_risk_reduction: float = SHIELD_DEFAULT_RISK_REDUCTION,
        allowed_interventions: Optional[frozenset] = None,
        scenario_family: Optional[str] = None,
        warning_variant: str = "rsa_obs_s1",
        user_cfg: Optional[dict] = None,
        dtmb_dispatch_cfg=None,
        factor_mode: str = "G_THETA",
        predictor_mode: str = "P4",
    ) -> V2EpisodeState:
        """Initialize a new episode. Returns episode state.

        If scenario_family is set, uses generate_scenario() from
        scenario_families module instead of generate_lattice_v2().
        """
        if scenario_family is not None:
            from .scenario_families import generate_scenario
            gm, cfg, meta, _sc = generate_scenario(
                scenario_family, seed=seed, difficulty=difficulty,
                latent_mode=latent_mode, user_cfg=user_cfg)
            _gate_mode = _sc.gate_mode
        else:
            gm, cfg, meta = generate_lattice_v2(
                seed=seed, difficulty=difficulty, latent_mode=latent_mode)
            _gate_mode = "block_risky"

        # ── PRS-2: Session-level WorldWeights override ──
        # If user_cfg contains a world_weights_override, re-derive cost/risk
        # from features using those shared weights. This makes the feature→risk
        # mapping consistent across episodes within a session, enabling transfer.
        if (user_cfg and 'world_weights_override' in user_cfg
                and latent_mode and meta.cell_features is not None):
            from ..agents.cost_risk_model import WorldWeights
            ww_override = user_cfg['world_weights_override']
            if isinstance(ww_override, WorldWeights):
                H_g, W_g = gm.height, gm.width
                for r in range(H_g):
                    for c in range(W_g):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = meta.cell_features[r, c]
                        gm.true_cost[r, c] = ww_override.true_cost(z)
                        gm.true_risk[r, c] = ww_override.true_risk(z)
                meta.world_weights = ww_override

        H, W = gm.height, gm.width
        rng = np.random.default_rng(seed * 1000 + 1)
        # Use scenario-provided t_max when available, else compute from time_ratio
        if scenario_family is not None:
            t_max = cfg.max_steps
        else:
            t_max = max(int(time_ratio * meta.shortest_safe), meta.shortest_safe + 1)

        fb = FeatureBeliefMap(H, W, FEATURE_DIM)
        belief_cost = np.ones((H, W), dtype=np.float64)
        belief_cost[gm.cell_types == CellType.WALL] = 100.0
        belief_cost[gm.cell_types == CellType.LOCKED_DOOR] = 100.0
        passable = np.ones((H, W), dtype=bool)
        passable[gm.cell_types == CellType.WALL] = False
        passable[gm.cell_types == CellType.LOCKED_DOOR] = False

        # Model setup: legacy vs latent
        lp = None
        if latent_mode:
            lp = latent_predictor if latent_predictor is not None else LatentCostRiskHead(
                d=FEATURE_DIM, risk_supervision=risk_supervision)
            rh = lp.risk_head  # expose for warning system compatibility
        else:
            rh = risk_head if risk_head is not None else BayesianRiskHead(d=FEATURE_DIM)

        tutor = None
        if tutor_mode in ("time_aware", "warn_first"):
            tutor = TimeAwareDoorTutor(gm, meta)

        # Phase 1A: Always init RSA channel for ALL variants
        # For legacy_bias: RSA runs as shadow diagnostics, does not affect behavior
        _rsa_channel = RSAWarningChannel()
        _rsa_belief_state = RSABeliefState()  # uniform prior

        # Use GridMap's start/goal when available (DTMB-L uses non-standard positions)
        _start = getattr(gm, 'agent_start', (2, 1)) or (2, 1)
        _goal = getattr(gm, 'target_pos', (2, W - 2)) or (2, W - 2)
        state = V2EpisodeState(
            gridmap=gm, config=cfg, meta=meta,
            agent_pos=_start, goal=_goal,
            feature_belief=fb, risk_head=rh,
            belief_cost=belief_cost, passable=passable,
            t=0, t_max=t_max,
            tutor=tutor, tutor_mode=tutor_mode,
            warning_mode=warning_mode,
            lambda_lane_warn=lambda_lane_warn,
            closure_budget=closure_budget,
            latent_mode=latent_mode,
            latent_predictor=lp,
            patch_radius=patch_radius,
            prefix_horizon=prefix_horizon,
            belief_planning_mode=belief_planning_mode,
            confidence_temperature=confidence_temperature,
            robot_belief_mode=robot_belief_mode,
            belief_copy_mode=belief_copy_mode,
            budget_mismatch=budget_mismatch,
            intervention_family_mode=intervention_family_mode,
            item_drop_enabled=item_drop_enabled,
            inventory=InventoryState(
                shield_risk_reduction=shield_risk_reduction,
            ) if intervention_family_mode else None,
            allowed_interventions=allowed_interventions,
            gate_mode=_gate_mode,
            warning_variant=warning_variant,
            rsa_channel=_rsa_channel,
            rsa_belief_state=_rsa_belief_state,
            rng=rng,
        )

        # Init robot belief surrogate if mode enabled
        if robot_belief_mode and lp is not None:
            state.robot_belief = init_robot_belief(
                fb.mean, fb.var, latent_predictor=lp,
                copy_mode=belief_copy_mode,
                budget_mismatch=budget_mismatch,
                rng=rng,
            )
            # Phase 10: Init Tutor Perceptual Model
            state.perceptual_access = init_perceptual_access(
                H, W, patch_radius=patch_radius)

        state.scenario_family = scenario_family
        state.dtmb_dispatch_cfg = dtmb_dispatch_cfg
        state.factor_mode = factor_mode
        state.predictor_mode = predictor_mode

        # GTET-L: init joint posterior for factor ablation / predictor audit
        if scenario_family == "goal_preference_temptation_entanglement_lattice":
            from ..teachers.joint_goal_pref_posterior import (
                JointGoalPrefPosterior, THETA_2, DEFAULT_TEMPT_GRID, DEFAULT_TEMPT_PRIOR,
            )
            state.gtet_posterior = JointGoalPrefPosterior(
                pref_types=THETA_2,
                tempt_grid=DEFAULT_TEMPT_GRID,
                tempt_prior=DEFAULT_TEMPT_PRIOR,
            )

        return state

    def step(self, s: V2EpisodeState) -> V2EpisodeState:
        """Advance one timestep: observe → tutor → plan → move → outcome.

        This is the semantic source of truth. step_teacher + step_agent
        on the env layer must compose to the exact same result.

        Mutates and returns the same state object for efficiency.
        """
        if s.done:
            return s
        self.observe(s)
        self.apply_tutor(s)
        self.plan_and_move(s)
        return s

    def observe(self, s: V2EpisodeState) -> None:
        """Sub-step 1: Agent observes features at current position.

        Uses patch observation when patch_radius > 1.
        Updates feature_belief (with visit metadata) and cue_cells_seen.
        No side effects on position, tutor state, or terminal flags.
        """
        if s.patch_radius > 1:
            fobs = observe_features_patch(
                s.agent_pos, s.meta.cell_features, s.gridmap.cell_types,
                patch_radius=s.patch_radius, rng=s.rng)
        else:
            fobs = observe_features(
                s.agent_pos, s.meta.cell_features, s.gridmap.cell_types,
                self_noise_var=0.01, neighbor_noise_var=0.08, rng=s.rng)
        for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
            s.feature_belief.update(pos[0], pos[1], f_obs, f_var, t=s.t)
            if s.gridmap.cell_types[pos[0], pos[1]] == CellType.RISKY:
                s.cue_cells_seen += 1

        # Phase 10: Update tutor's perceptual access model
        if s.perceptual_access is not None:
            update_perceptual_access(s.perceptual_access, s.agent_pos, s.passable)

        # Sync robot belief surrogate (Phase 7)
        if s.robot_belief_mode and s.robot_belief is not None:
            sync_robot_belief(
                s.robot_belief, s.feature_belief.mean, s.feature_belief.var,
                latent_predictor=s.latent_predictor, t=s.t, rng=s.rng,
            )

    def apply_tutor(self, s: V2EpisodeState) -> None:
        """Sub-step 2: Teacher actions (door closing / warnings).

        Updates passable, belief_cost, warned_*, unlock_count, warn_count.
        Does NOT move the agent or update terminal flags.
        """
        self._apply_tutor_dispatch(s)

    def plan_and_move(self, s: V2EpisodeState) -> None:
        """Sub-step 3: Agent plans, moves, and resolves outcome.

        Updates agent_pos, steps, risk_head, risky_entered, traps_hit,
        survived, reached_goal, done. Optionally computes prefix/belief diagnostics.
        """
        extra = s.warned_cell_extra if s.warned_cell_extra else None

        # Compute route necessity: shared scalar for the entire plan.
        # How critical is traversing unvisited territory for reaching goal?
        from ..agents.route_necessity import compute_route_necessity
        if s.latent_predictor is not None and s.t_max > 0:
            # Find cells the agent hasn't visited — "unknown territory"
            unvisited = set()
            for r in range(s.gridmap.height):
                for c in range(s.gridmap.width):
                    if s.passable[r, c] and not s.feature_belief.memory[r, c].ever_traversed:
                        unvisited.add((r, c))
            route_necessity = compute_route_necessity(
                s.agent_pos, s.goal, s.passable, s.t, s.t_max,
                route_cells=unvisited,
            )
        else:
            route_necessity = 0.0

        if s.belief_planning_mode and s.latent_predictor is not None:
            # Phase 6: belief-conditioned bounded planning
            bp = plan_from_belief(
                s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
                s.risk_head, s.passable,
                latent_predictor=s.latent_predictor,
                warned_cell_extra=extra,
                search_budget=30,
                prefix_horizon=max(s.prefix_horizon, 5),
                confidence_temperature=s.confidence_temperature,
                t=s.t, t_max=s.t_max,
                feature_belief_var=s.feature_belief.var,
                route_necessity=route_necessity,
            )
            next_pos = bp.next_pos
            path = bp.full_path
            s.last_belief_plan = bp
            s.last_prefix = bp.prefix_prediction

            # Failure modes
            from ..agents.planner_astar import plan_with_alternatives_v2
            _, _, _, cand_scores = plan_with_alternatives_v2(
                s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
                s.risk_head, budget=30, passable_mask=s.passable,
                warned_cell_extra_cost=extra,
                latent_predictor=s.latent_predictor,
                feature_belief_var=s.feature_belief.var,
                route_necessity=route_necessity)
            warned_set = set(s.warned_cell_extra.keys()) if s.warned_cell_extra else set()
            s.last_failure_modes = estimate_failure_modes(
                bp, s.t, s.t_max, cand_scores, warned_cells=warned_set)
        else:
            # Legacy / latent / patch path
            _, next_pos, path = plan_next_action_v2(
                s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
                s.risk_head, budget=30, passable_mask=s.passable,
                warned_cell_extra_cost=extra,
                latent_predictor=s.latent_predictor,
                feature_belief_var=s.feature_belief.var,
                route_necessity=route_necessity)

            # Prefix prediction (read-only diagnostics)
            if s.prefix_horizon > 0 and s.latent_predictor is not None and path:
                s.last_prefix = compute_prefix_predictions(
                    path, s.feature_belief.mean, s.latent_predictor,
                    horizon=s.prefix_horizon)
            else:
                s.last_prefix = None
            s.last_belief_plan = None
            s.last_failure_modes = None

        s.agent_pos = next_pos
        s.steps += 1
        s.feature_belief.mark_traversed(next_pos[0], next_pos[1], s.t)

        # ── Outcome ──
        r, c = s.agent_pos
        x_belief = s.feature_belief.get_mean(r, c)
        true_cost = float(s.gridmap.true_cost[r, c]) if hasattr(s.gridmap, 'true_cost') else 1.0
        true_risk_val = float(s.gridmap.true_risk[r, c])

        if s.gridmap.cell_types[r, c] == CellType.RISKY:
            s.risky_entered += 1
            effective_risk = true_risk_val
            # Shield consumption (Phase 8): same reduction as planner
            if s.inventory is not None and s.inventory.has_shield():
                effective_risk *= (1.0 - s.inventory.shield_risk_reduction)
                s.inventory.consume_shield()
            if s.rng.random() < effective_risk:
                s.traps_hit += 1
                s.survived = False
                if s.latent_predictor:
                    s.latent_predictor.update_from_outcome(
                        x_belief, cost_label=true_cost, risk_label=1.0, weight=4.0)
                else:
                    s.risk_head.update_from_label(x_belief, 1.0, weight=4.0)
                s.done = True
                return
            else:
                if s.latent_predictor:
                    risk_label = true_risk_val if s.latent_predictor.risk_supervision == "oracle_visited" else 0.0
                    s.latent_predictor.update_from_outcome(
                        x_belief, cost_label=true_cost, risk_label=risk_label, weight=1.5)
                else:
                    s.risk_head.update_from_label(x_belief, true_risk_val, weight=1.5)
        else:
            if s.latent_predictor:
                s.latent_predictor.update_from_outcome(
                    x_belief, cost_label=true_cost, risk_label=0.0, weight=0.1)
            else:
                s.risk_head.update_from_label(x_belief, 0.0, weight=0.1)

        if s.agent_pos == s.goal:
            s.reached_goal = True
            s.done = True
        elif s.t + 1 >= s.t_max:
            s.done = True
        else:
            s.t += 1

        # Phase 0 audit: per-step uncertainty trace (read-only)
        if s.audit_mode and s.latent_predictor is not None:
            _obs_cells = []
            for dr in range(-max(s.patch_radius, 1), max(s.patch_radius, 1) + 1):
                for dc in range(-max(s.patch_radius, 1), max(s.patch_radius, 1) + 1):
                    _ar, _ac = s.agent_pos[0] + dr, s.agent_pos[1] + dc
                    if 0 <= _ar < s.gridmap.height and 0 <= _ac < s.gridmap.width:
                        if s.passable[_ar, _ac]:
                            _obs_cells.append((_ar, _ac))
            if _obs_cells:
                _u_sum = 0.0
                for _cr, _cc in _obs_cells:
                    _xb = s.feature_belief.get_mean(_cr, _cc)
                    _uc = s.latent_predictor.predict_cost_uncertainty(_xb)
                    _ur = s.latent_predictor.predict_risk_uncertainty(_xb)
                    _u_sum += _uc + _ur
                _U_t = _u_sum / len(_obs_cells)
            else:
                _U_t = 0.0
            _tutor_act = s.last_intervention.action if s.last_intervention else "NONE"
            s.audit_trace.append({
                "t": s.t,
                "U_t": _U_t,
                "action": _tutor_act,
                "pos": s.agent_pos,
                "cost": true_cost,
            })

    def _apply_tutor_dispatch(self, s: V2EpisodeState) -> None:
        """Dispatch tutor/warning actions based on mode."""

        if s.tutor_mode in ("time_aware", "warn_first") and s.tutor:
            # Skip close_risky_gate for unlock_shortcut families —
            # gate should be OPENED, not closed
            if s.gate_mode == "unlock_shortcut":
                pass  # Do not invoke legacy gate-closing tutor
            else:
                budget_ok = (s.closure_budget is None) or (s.unlock_count < s.closure_budget)
                if budget_ok:
                    actions = s.tutor.step(s.agent_pos, s.t_max - s.t, s.t)
                    for a in actions:
                        if a.action == "close_risky_gate":
                            if s.closure_budget is not None and s.unlock_count >= s.closure_budget:
                                break
                            s.passable[a.gate_cell] = False
                            s.belief_cost[a.gate_cell] = 100.0
                            s.unlock_count += 1
                        elif a.action == "warn_only":
                            self._apply_segment_warning(s, a.segment_index)

        elif s.tutor_mode == "always_close":
            if s.t == 0:
                for seg in s.meta.segments:
                    if s.closure_budget is not None and s.unlock_count >= s.closure_budget:
                        break
                    s.passable[seg.risky_entry_gate] = False
                    s.belief_cost[seg.risky_entry_gate] = 100.0
                    s.unlock_count += 1
                                        
        elif s.tutor_mode == "dtmb_oracle":
            if getattr(s, 'scenario_family', None) == "deep_tree_mixed_bottleneck_lattice":
                from .dtmb_helpers import apply_dtmb_oracle_action
                _dcfg = getattr(s, 'dtmb_dispatch_cfg', None)
                apply_dtmb_oracle_action(s, dispatch_cfg=_dcfg)
            return

        # Warning-only mode (no door tutor)
        if s.warning_mode in ("fixed", "selected") and s.tutor_mode == "none":
            if s.agent_pos[0] == 2:
                for seg in s.meta.segments:
                    if seg.index in s.warned_segments:
                        continue
                    if abs(s.agent_pos[1] - seg.col_start) > 2:
                        continue
                    # Route through _apply_segment_warning for variant dispatch
                    self._apply_segment_warning(s, seg.index)

        # Robot-belief intervention mode (Phase 7+8)
        if s.robot_belief_mode and s.robot_belief is not None and s.latent_predictor is not None:
            extra = s.warned_cell_extra if s.warned_cell_extra else None
            icfg = InterventionConfig(
                item_drop_enabled=(s.item_drop_enabled and s.intervention_family_mode),
                boredom_weight=s.boredom_weight,
            )

            # ── GTET-L: always update posterior for all modes (including FULL) ──
            if getattr(s, 'gtet_posterior', None) is not None:
                _simulate_gtet_posterior_update(s)

            # ── GTET-L factor ablation: modify intervention weights (non-FULL only) ──
            if (getattr(s, 'factor_mode', 'FULL') != 'FULL'
                    and getattr(s, 'gtet_posterior', None) is not None):
                icfg = _apply_gtet_factor_modifier(s, icfg)

            decision = score_interventions(
                s.robot_belief, s.agent_pos, s.goal,
                s.belief_cost, s.passable, s.meta,
                warned_cell_extra=extra,
                warned_segments=s.warned_segments,
                prefix_horizon=max(s.prefix_horizon, 5),
                t=s.t, t_max=s.t_max,
                config=icfg,
                inventory_state=s.inventory,
                allowed_actions=s.allowed_interventions,
                perceptual_access=s.perceptual_access,
            )
            s.last_intervention = decision

            # Execute the chosen intervention
            if decision.action == "WARN":
                if getattr(s, 'scenario_family', None) == "deep_tree_mixed_bottleneck_lattice":
                    from .dtmb_helpers import apply_dtmb_warning
                    _wv = getattr(getattr(s, 'dtmb_dispatch_cfg', None), 'warn_variant', 'W1') or 'W1'
                    apply_dtmb_warning(s, variant=_wv)
                elif getattr(s, 'scenario_family', None) == "goal_preference_temptation_entanglement_lattice":
                    # GTET-L WARN: one-shot, posterior-guided
                    if s.warn_count == 0:  # one-shot: first warning is irrevocable
                        _apply_gtet_warning(s)
                else:
                    for seg in s.meta.segments:
                        if seg.index in s.warned_segments:
                            continue
                        if seg.col_start > s.agent_pos[1]:
                            self._apply_segment_warning(s, seg.index)
                            break  # one warning per step
            elif decision.action == "UNLOCK":
                # Try all_door_positions first (for unlock_shortcut families)
                unlocked = False
                if s.gate_mode == "unlock_shortcut":
                    for door_pos in s.meta.all_door_positions:
                        r, c = door_pos
                        if not s.passable[r, c]:
                            s.passable[r, c] = True
                            s.belief_cost[r, c] = 1.0
                            s.unlock_count += 1
                            # Phase 10: UNLOCK reduces uncertainty on newly opened cell
                            s.feature_belief.apply_unlock_update(r, c, t=s.t)
                            unlocked = True
                            break  # one unlock per step
                if not unlocked:
                    # Legacy: open risky_entry_gate
                    for seg in s.meta.segments:
                        gate = seg.risky_entry_gate
                        if not s.passable[gate[0], gate[1]] and gate[1] > s.agent_pos[1]:
                            s.passable[gate[0], gate[1]] = True
                            s.belief_cost[gate[0], gate[1]] = 1.0
                            s.unlock_count += 1
                            # Phase 10: UNLOCK reduces uncertainty
                            s.feature_belief.apply_unlock_update(gate[0], gate[1], t=s.t)
                            break  # one unlock per step
            elif decision.action == "ITEM_DROP":
                # ITEM_DROP only changes traversal dynamics — NOT belief
                if s.inventory is not None:
                    s.inventory.add_shield()
            # WAIT: no action
        else:
            s.last_intervention = None

    @staticmethod
    def _compute_branch_nll(
        seg, feature_belief, risk_head,
        warned_cell_extra: dict,
        theta: str = "safe",
    ) -> float:
        """Compute NLL of safe-branch choice from segment-level BranchAttributes.

        Builds two BranchAttributes (risky_branch, safe_branch) from segment
        topology + current belief, then returns -log P(safe | theta, branches).

        Used for ΔNLL_local: NLL_before - NLL_after should be negative
        if warning correctly increases P(safe).
        """
        # Risky branch: mean risk from belief over risky cells
        risky_risk_scores = []
        for rc in seg.risky_cells:
            x = feature_belief.get_mean(rc[0], rc[1])
            base_risk = float(risk_head.predict_risk(x))
            extra = warned_cell_extra.get(rc, 0.0)
            risky_risk_scores.append(base_risk + extra * 0.1)  # scale extra
        mean_risky_risk = np.mean(risky_risk_scores) if risky_risk_scores else 0.3

        # Safe branch: mean risk from safe cells (should be low)
        safe_risk_scores = []
        for rc in seg.safe_cells[:5]:  # cap to avoid slow loops
            x = feature_belief.get_mean(rc[0], rc[1])
            safe_risk_scores.append(float(risk_head.predict_risk(x)))
        mean_safe_risk = np.mean(safe_risk_scores) if safe_risk_scores else 0.05

        branches = [
            BranchAttributes(
                safety_score=1.0 - mean_risky_risk,
                risk_penalty=mean_risky_risk,
            ),
            BranchAttributes(
                safety_score=1.0 - mean_safe_risk,
                risk_penalty=mean_safe_risk,
            ),
        ]
        params = AgentPolicyParams()
        probs = compute_choice_probs(branches, theta, params)
        # NLL of safe branch choice (index 1)
        return -float(np.log(probs[1] + 1e-10))

    def _apply_segment_warning(self, s: V2EpisodeState, seg_index: int) -> None:
        """Apply warning to a specific segment — Phase 1A unified routing.

        ALL variants now route through compute_warning_belief_delta() as the
        single semantic source. The variant determines which adapters fire:

          legacy_bias:
            - RSA runs as shadow diagnostics (b⁺ computed but NOT used)
            - Legacy apply_warning() fires as before (pseudo-label + lane bias)
            - Phase 10 apply_warn_update fires as before
            - Behavior: IDENTICAL to pre-Phase-1A

          rsa_obs_l0 / rsa_obs_s1 / rsa_obs_s1_trust:
            - RSA belief update → planner adapter → warned_cell_extra
            - NO pseudo-label, NO lane bias, NO Phase 10 apply_warn_update

          rsa_plus_phase10:
            - RSA belief update + planner adapter
            - ALSO applies legacy pseudo-label + Phase 10 (hybrid ablation)
        """
        if seg_index in s.warned_segments:
            return
        seg = s.meta.segments[seg_index]

        # 1. Select utterance (same action-gap logic for all variants)
        utt = select_best_warning_action_gap(
            seg.risky_cells, s.feature_belief, s.risk_head,
            lambda_lane_warn=s.lambda_lane_warn)
        if utt is None:
            utt = Utterance.RISKY_TEXTURE_AHEAD

        # 2. Map segment → RSA context + utterance
        rsa_ctx = map_segment_to_rsa_context(seg)
        risky_side = rsa_ctx["risky_side"]
        rsa_utt = s.rsa_channel.select_utterance(
            true_risk_side=risky_side,
            context=rsa_ctx,
            prior=s.rsa_belief_state.belief,
        )

        # 3. Snapshot NLL before warning
        nll_before = self._compute_branch_nll(
            seg, s.feature_belief, s.risk_head, s.warned_cell_extra)

        # 4. Determine RSA variant and adapter flags
        rsa_mode_map = {
            "legacy_bias": "s1",           # shadow: compute but don't use
            "rsa_obs_l0": "l0",
            "rsa_obs_s1": "s1",
            "rsa_obs_s1_trust": "s1_trust",
            "rsa_plus_phase10": "s1",
        }
        rsa_variant = rsa_mode_map.get(s.warning_variant, "s1")

        # 5. Compute unified WarningBeliefDelta (single semantic source)
        #    For legacy_bias: uses a COPY of belief state (shadow only)
        if s.warning_variant == "legacy_bias":
            # Shadow: compute RSA delta on a copy, don't mutate real state
            from copy import deepcopy
            shadow_belief = deepcopy(s.rsa_belief_state)
            delta = compute_warning_belief_delta(
                rsa_channel=s.rsa_channel,
                belief_state=shadow_belief,
                utterance=rsa_utt,
                context=rsa_ctx,
                segment_risky_cells=seg.risky_cells,
                segment_side=risky_side,
                lambda_lane_warn=s.lambda_lane_warn,
                variant=rsa_variant,
                enable_pseudolabel=False,
            )
            # Update shadow belief state for diagnostics only
            s.rsa_belief_state = shadow_belief
        else:
            # Active: compute RSA delta and mutate real belief state
            enable_pseudo = (s.warning_variant == "rsa_plus_phase10")
            delta = compute_warning_belief_delta(
                rsa_channel=s.rsa_channel,
                belief_state=s.rsa_belief_state,
                utterance=rsa_utt,
                context=rsa_ctx,
                segment_risky_cells=seg.risky_cells,
                segment_side=risky_side,
                lambda_lane_warn=s.lambda_lane_warn,
                variant=rsa_variant,
                enable_pseudolabel=enable_pseudo,
                feature_belief=s.feature_belief if enable_pseudo else None,
            )

        # 6. Apply downstream adapters
        if s.warning_variant == "legacy_bias":
            # ── Legacy path: original behavior preserved ──
            apply_warning(utt, seg.risky_cells, s.feature_belief, s.risk_head,
                          s.warned_lane_bias, seg_index,
                          weight=5.0, tau=0.3,
                          lambda_lane_warn=s.lambda_lane_warn)

            # Phase 10: WARN as belief evidence factor
            warn_dir = None
            if s.latent_predictor is not None and hasattr(s.latent_predictor, 'risk_head'):
                w_r = s.latent_predictor.risk_head.w
                w_norm = float(np.linalg.norm(w_r))
                if w_norm > 1e-6:
                    warn_dir = w_r / w_norm
            for rc in seg.risky_cells:
                s.feature_belief.apply_warn_update(
                    rc[0], rc[1], warn_direction=warn_dir,
                    warn_strength=0.15, warn_confidence=2.0)

            s.warned_cell_extra = _build_warned_cell_extra(
                s.warned_lane_bias, s.meta)

        elif s.warning_variant in ("rsa_obs_l0", "rsa_obs_s1", "rsa_obs_s1_trust"):
            # ── RSA-only: planner adapter, no pseudo-labels ──
            apply_planner_adapter(delta, s.warned_cell_extra)

        elif s.warning_variant == "rsa_plus_phase10":
            # ── Hybrid: RSA planner adapter + legacy pseudo-labels ──
            apply_planner_adapter(delta, s.warned_cell_extra)
            apply_pseudolabel_adapter(delta, s.risk_head)
            # Phase 10: WARN as belief evidence factor
            warn_dir = None
            if s.latent_predictor is not None and hasattr(s.latent_predictor, 'risk_head'):
                w_r = s.latent_predictor.risk_head.w
                w_norm = float(np.linalg.norm(w_r))
                if w_norm > 1e-6:
                    warn_dir = w_r / w_norm
            for rc in seg.risky_cells:
                s.feature_belief.apply_warn_update(
                    rc[0], rc[1], warn_direction=warn_dir,
                    warn_strength=0.15, warn_confidence=2.0)

        # 7. Update state
        s.warned_segments.add(seg_index)
        s.warn_count += 1

        # 8. ΔNLL_local diagnostics
        nll_after = self._compute_branch_nll(
            seg, s.feature_belief, s.risk_head, s.warned_cell_extra)
        delta_nll_local = nll_after - nll_before

        diag = {
            "variant": s.warning_variant,
            "segment_index": seg_index,
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "delta_nll_local": round(delta_nll_local, 4),
            # Phase 1A unified fields
            "delta_rho_inc": delta.delta_rho_inc,
            "delta_rho_uniform": delta.delta_rho_uniform,
            "rsa_utterance": delta.utterance,
            "risky_side": risky_side,
            "rsa_entropy_after": delta.diagnostics.get("entropy_after", 0),
            "belief_posterior": delta.posterior_belief.tolist(),
        }
        s.rsa_warn_diagnostics.append(diag)

    @staticmethod
    def get_metrics(s: V2EpisodeState) -> dict:
        """Extract result metrics dict (compatible with old run_episode output)."""
        return {
            "survived": s.survived,
            "reached_goal": s.reached_goal,
            "steps": s.steps,
            "t_max": s.t_max,
            "traps": s.traps_hit,
            # Canonical names
            "risky_entered": s.risky_entered,
            "unlock_count": s.unlock_count,
            "warn_count": s.warn_count,
            "cue_cells_seen": s.cue_cells_seen,
            # Backward-compatible aliases
            "risky": s.risky_entered,
            "closures": s.unlock_count,
            "warnings": s.warn_count,
            "cue_seen": s.cue_cells_seen,
        }

    @staticmethod
    def get_extended_metrics(s: V2EpisodeState) -> dict:
        """Extended metrics for Phase 9+ evaluation. Additive, non-breaking."""
        base = LatticeV2Runner.get_metrics(s)
        base.update({
            "success": s.reached_goal and s.survived,
            "death": not s.survived,
            "timeout": not s.reached_goal and s.survived,
            "cumulative_cost": float(s.steps),
            "cumulative_risk": float(s.risky_entered),
            "intervention_count": s.warn_count + s.unlock_count,
            "has_inventory": s.inventory is not None,
            "shield_remaining": s.inventory.shield if s.inventory else 0,
        })
        if s.last_intervention is not None:
            base["last_intervention_action"] = s.last_intervention.action
            base["last_intervention_scores"] = s.last_intervention.scores

        # Step 2: RSA warning diagnostics
        base["warning_variant"] = s.warning_variant
        if s.rsa_belief_state is not None:
            base["rsa_belief_final"] = s.rsa_belief_state.belief.tolist()
            base["rsa_entropy_final"] = s.rsa_belief_state.entropy()
            base["rsa_n_updates"] = s.rsa_belief_state.n_updates
        if s.rsa_warn_diagnostics:
            base["rsa_warn_diagnostics"] = s.rsa_warn_diagnostics

        return base


# ══════════════════════════════════════════════════════════════════════
# GTET-L Factor Ablation Hook (external to class, called from dispatch)
# ══════════════════════════════════════════════════════════════════════

def _apply_gtet_factor_modifier(
    s,  # V2EpisodeState
    icfg,  # InterventionConfig
):
    """Modify InterventionConfig based on factor-restricted posterior.

    The GTET posterior is updated from the agent's current position (as a
    proxy for observed branch choices), then the factor-restricted view is
    compared against the full view to derive epistemic/risk modifiers.

    Modifies these InterventionConfig weights:
      - warn_effect_weight:  scaled by epistemic modifier (KL-based)
      - catastrophe_weight:  biased by temptation risk modifier
    """
    from ..teachers.gtet_factor_adapter import (
        build_factor_restricted_view,
        compute_posterior_epistemic_modifier,
        compute_posterior_risk_modifier,
    )
    from ..teachers.joint_goal_pref_posterior import DEFAULT_TEMPT_GRID
    from copy import copy

    jgpp = s.gtet_posterior
    if jgpp is None:
        return icfg

    # Posterior already updated upstream in apply_tutor dispatch

    # Build full and restricted views
    q_full = jgpp._weights()
    q_restricted = build_factor_restricted_view(jgpp, s.factor_mode)

    # Compute modifiers
    kl_div = compute_posterior_epistemic_modifier(q_full, q_restricted)
    risk_bias = compute_posterior_risk_modifier(
        q_restricted, DEFAULT_TEMPT_GRID)

    # Create modified config (don't mutate original)
    icfg_new = copy(icfg)

    # Epistemic modifier: when factor mode drops important dims,
    # KL is high -> reduce warn effectiveness (tutor has less info)
    if kl_div > 0.01:
        icfg_new.warn_effect_weight *= max(0.3, 1.0 - kl_div * 0.5)
        icfg_new.learning_gain_weight *= max(0.3, 1.0 - kl_div * 0.3)

    # Risk modifier: high temptation MAP -> boost catastrophe weight
    if risk_bias > 0.02:
        icfg_new.catastrophe_weight *= (1.0 + risk_bias * 2.0)

    # Log for ADR computation
    s.gtet_action_log.append({
        "t": s.t,
        "factor_mode": s.factor_mode,
        "kl_div": float(kl_div),
        "risk_bias": float(risk_bias),
        "warn_weight": float(icfg_new.warn_effect_weight),
        "cat_weight": float(icfg_new.catastrophe_weight),
    })

    return icfg_new


def _simulate_gtet_posterior_update(s):
    """Update the GTET posterior from agent position using GTET cue metadata.

    Creates genuine g×θ×z correlations by:
    1. Reading which cue cells the agent is near/has passed through
    2. Constructing branch observations that reflect the cue information
    3. Updating posterior so that marginalization loses information
    4. [NEW] Direct z-cue likelihood: temptation cues boost high-z hypotheses

    Only updates if the agent's position is near cue cells.
    """
    jgpp = s.gtet_posterior
    if jgpp is None or s.t == 0:
        return

    from ..agents.stochastic_agent_policy import BranchAttributes
    import numpy as np
    r, c = s.agent_pos
    H = s.gridmap.height
    center = H // 2
    row_offset = r - center

    # Check if near GTET cue cells for stronger updates
    gtet_meta = getattr(s.meta, 'gtet_meta', None)
    near_goal_cue = False
    near_tempt_cue = False
    tempt_cue_intensity = 0.0
    if gtet_meta is not None:
        gt = gtet_meta.goal_cue_tags
        tt = gtet_meta.temptation_cue_tags
        # Check 3x3 neighborhood
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < s.gridmap.width:
                    if gt[nr, nc] >= 0:
                        near_goal_cue = True
                    if tt[nr, nc] > 0.1:
                        near_tempt_cue = True
                        tempt_cue_intensity = max(tempt_cue_intensity,
                                                   float(tt[nr, nc]))

    # Build branches with CUE-CORRELATED attributes
    if near_goal_cue and near_tempt_cue:
        branches = [
            BranchAttributes(safety_score=0.9, temptation_score=0.1,
                             texture_novelty=0.2, shortcut_bonus=0.0,
                             risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.8,
                             texture_novelty=0.7, shortcut_bonus=0.2,
                             risk_penalty=0.5),
        ]
    elif near_goal_cue:
        branches = [
            BranchAttributes(safety_score=0.85, temptation_score=0.15,
                             texture_novelty=0.15, shortcut_bonus=0.0,
                             risk_penalty=0.1),
            BranchAttributes(safety_score=0.5, temptation_score=0.4,
                             texture_novelty=0.3, shortcut_bonus=0.1,
                             risk_penalty=0.3),
        ]
    elif near_tempt_cue:
        branches = [
            BranchAttributes(safety_score=0.7, temptation_score=0.3,
                             texture_novelty=0.2, shortcut_bonus=0.0,
                             risk_penalty=0.15),
            BranchAttributes(safety_score=0.3, temptation_score=0.7,
                             texture_novelty=0.6, shortcut_bonus=0.15,
                             risk_penalty=0.45),
        ]
    else:
        branches = [
            BranchAttributes(safety_score=0.65, temptation_score=0.25,
                             texture_novelty=0.2, shortcut_bonus=0.0,
                             risk_penalty=0.15),
            BranchAttributes(safety_score=0.55, temptation_score=0.35,
                             texture_novelty=0.3, shortcut_bonus=0.05,
                             risk_penalty=0.2),
        ]

    # Observed action: 0=safe branch (center), 1=risky branch (offset)
    observed = 1 if abs(row_offset) > 1 else 0

    try:
        jgpp.update(None, branches, observed)
    except Exception:
        pass  # Don't crash on posterior update failure






def _apply_gtet_warning(s):
    """Apply GTET-L fair posterior-guided warning with selectable predictor.

    FAIR DISPATCH CONTRACT (unchanged):
    - ALL factor modes use the SAME dispatch logic
    - Warning budget = 1 belt-zone
    - No omniscient fallback

    Predictor modes (s.predictor_mode):
        P1: E[z]-based (current baseline, known to be biased)
        P2: Joint MAP (g*,θ*,z*) → single hypothesis route prediction
        P3: Route posterior mixture — marginalize P(upper|g,θ,z) over q
        P4: z-masked route mixture — same but only using q(g,θ)
    """
    agent_col = s.agent_pos[1]
    warned_any = False
    H = s.gridmap.height
    center = H // 2

    # ── Step 1: Collect belt cells by zone ──
    upper_belts = []
    lower_belts = []
    if hasattr(s.meta, 'belt_cells_by_stage') and s.meta.belt_cells_by_stage:
        for stage_belts in s.meta.belt_cells_by_stage:
            for r, c in stage_belts:
                if c > agent_col and (r, c) not in s.warned_cell_extra:
                    if r <= center:
                        upper_belts.append((r, c))
                    else:
                        lower_belts.append((r, c))

    if not upper_belts and not lower_belts:
        return  # nothing to warn about

    # ── Step 2: Compute P(upper) using selected predictor ──
    jgpp = getattr(s, 'gtet_posterior', None)
    factor_mode = getattr(s, 'factor_mode', 'FULL')
    predictor = getattr(s, 'predictor_mode', 'P1')

    from ..teachers.joint_goal_pref_posterior import (
        DEFAULT_TEMPT_GRID, DEFAULT_TEMPT_PRIOR,
    )

    if jgpp is not None:
        from ..teachers.gtet_factor_adapter import build_factor_restricted_view
        q_view = build_factor_restricted_view(jgpp, factor_mode)
        p_upper = _compute_p_upper(q_view, predictor, DEFAULT_TEMPT_GRID)
    else:
        # No posterior → prior E[z]
        tp = list(DEFAULT_TEMPT_PRIOR) if DEFAULT_TEMPT_PRIOR else [1.0] * len(DEFAULT_TEMPT_GRID)
        tp_sum = sum(tp)
        expected_z = sum(tp[i] / tp_sum * DEFAULT_TEMPT_GRID[i]
                         for i in range(len(DEFAULT_TEMPT_GRID)))
        p_upper = min(0.95, max(0.05, expected_z / 0.9))

    p_lower = 1.0 - p_upper

    # ── Step 3: Score each belt zone ──
    def _zone_risk(cells):
        if not cells:
            return 0.0
        risks = []
        for r, c in cells:
            risks.append(s.gridmap.risk[r, c] if hasattr(s.gridmap, 'risk')
                         else 0.45)
        return sum(risks) / len(risks)

    def _zone_urgency(cells):
        if not cells:
            return 0.0
        min_dist = min(abs(c - agent_col) for _, c in cells)
        return 1.0 / (1.0 + min_dist)

    s_upper = p_upper * _zone_risk(upper_belts) * _zone_urgency(upper_belts)
    s_lower = p_lower * _zone_risk(lower_belts) * _zone_urgency(lower_belts)

    # ── Step 4: Warn highest-scoring zone ──
    if s_upper >= s_lower and upper_belts:
        chosen_belts = upper_belts
    elif lower_belts:
        chosen_belts = lower_belts
    else:
        chosen_belts = upper_belts

    for r, c in chosen_belts:
        s.warned_cell_extra[(r, c)] = 5.0
        s.belief_cost[r, c] += 5.0
        warned_any = True

    # ── Step 5: Temptation cue cells (same for all) ──
    if hasattr(s.meta, 'temptation_cue_cells') and s.meta.temptation_cue_cells:
        for r, c in s.meta.temptation_cue_cells:
            if c > agent_col and (r, c) not in s.warned_cell_extra:
                s.warned_cell_extra[(r, c)] = 3.0
                s.belief_cost[r, c] += 3.0
                warned_any = True

    if warned_any:
        s.warn_count += 1


def _compute_p_upper(q_view, predictor, tempt_grid):
    """Compute P(agent takes upper/risky route) using different predictors.

    Args:
        q_view: Posterior view, shape (n_g, n_p, n_z), normalized.
        predictor: "P1" | "P2" | "P3" | "P4"
        tempt_grid: Tuple of z values, e.g. (0.0, 0.3, 0.6, 0.9)

    Returns:
        p_upper in [0.05, 0.95]
    """
    import numpy as np
    n_g, n_p, n_z = q_view.shape

    if predictor == "P1":
        # ── P1: E[z]-based ──
        marg_z = q_view.sum(axis=(0, 1))  # shape (n_z,)
        expected_z = sum(marg_z[i] * tempt_grid[i] for i in range(n_z))
        p_upper_raw = expected_z / 0.9

    elif predictor == "P2":
        # ── P2: Joint MAP ──
        flat_idx = np.argmax(q_view)
        gi, pi, zi = np.unravel_index(flat_idx, q_view.shape)
        z_map = tempt_grid[zi]
        p_upper_raw = z_map / 0.9

    elif predictor == "P3":
        # ── P3: Route posterior mixture (full joint) ──
        # P(upper | q) = Σ_{g,θ,z} P(upper | g,θ,z) * q(g,θ,z)
        # P(upper | g,θ,z) ≈ z/0.9 (higher temptation → upper route)
        # This marginalizes over ALL latent factors
        p_upper_raw = 0.0
        for gi in range(n_g):
            for pi in range(n_p):
                for zi in range(n_z):
                    w = q_view[gi, pi, zi]
                    if w < 1e-12:
                        continue
                    z_val = tempt_grid[zi]
                    # Route preference: high z → upper, low z → lower
                    # Also use goal/pref: goal 0 = upper-path goal, etc.
                    # Simplified: z dominates route choice
                    p_upper_given_h = z_val / 0.9
                    p_upper_raw += w * p_upper_given_h

    elif predictor == "P4":
        # ── P4: z-masked route mixture ──
        # Marginalize out z first, then predict
        # P(upper | q_{g,θ}) = Σ_{g,θ} P(upper | g,θ) * q(g,θ)
        # P(upper | g,θ) = 0.5 (no z info → no preference)
        # This should give p_upper ≈ 0.5 always
        p_upper_raw = 0.5

    else:
        p_upper_raw = 0.5

    return min(0.95, max(0.05, p_upper_raw))
