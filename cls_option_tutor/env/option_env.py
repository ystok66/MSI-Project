"""
option_env.py — V2 block/query stepping logic.

V2 changes:
  - Risk-only refresh: text options fixed per query, only risk re-sampled
  - Discrete damage: damage = risk_class (deterministic)
  - HP_0 = 5
  - RISK_HINT support in tutor_act()
  - HIGHLIGHT persists through refresh
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..config import FullConfig, EnvConfig
from ..interfaces import Option, RevealEvent, LearnerStep, TutorStep, Example
from ..env.state import QueryState, BlockState, ProfileState
from ..env.danger_model import DangerModel, generate_danger_model
from ..env.interventions import (
    apply_wait, apply_ban, apply_highlight, apply_skip,
    apply_risk_hint, apply_shortlist, apply_mix,
    clear_menu_interventions, get_active_menu,
)
from ..grammar.task_adapter import TaskAdapter, Grammar
from ..grammar.option_generator import generate_menu
from ..grammar.option_generator_v2 import ProgramPool, generate_menu_v2
from ..grammar.option_generator_diagnostic import generate_menu_diagnostic


class OptionEnv:
    """V2 block-structured discrete option-selection environment.

    Key V2 semantics:
      - Menu text fixed per query; refresh only re-samples risk
      - damage = risk_class (deterministic)
      - HIGHLIGHT persists through refresh
    """

    def __init__(self, cfg: Optional[FullConfig] = None,
                 data_dir: str = ""):
        self.cfg = cfg or FullConfig()
        self.data_dir = data_dir or getattr(self.cfg, 'cls_data_dir', '')
        self.adapter = TaskAdapter(self.data_dir)
        self._danger_model: Optional[DangerModel] = None
        self._grammar: Optional[Grammar] = None
        self._support: List[Example] = []
        self._query_pool: List[Example] = []
        self._rng: Optional[np.random.Generator] = None
        self._pool: Optional[ProgramPool] = None  # V2 program pool

    # ── Reset ───────────────────────────────────────────────────

    def reset_block(
        self,
        task_id: str,
        seed: int = 42,
        block_id: int = 0,
        synthesize: bool = False,
    ) -> BlockState:
        """Initialize a new block from a CLS task file."""
        self._rng = np.random.default_rng(seed)
        support, queries_raw, grammar = self.adapter.load_task(task_id)

        self._grammar = grammar
        self._support = support
        self._query_pool = queries_raw

        # Generate danger model for this block
        self._danger_model = generate_danger_model(
            m=self.cfg.env.danger_dim,
            rng=self._rng,
            cluster_sigma=self.cfg.env.cluster_sigma,
        )

        # Build V2 program pool (all valid renderable programs)
        self._pool = ProgramPool(
            grammar, support,
            max_program_len=5, max_output_len=8,
        )

        # Select queries
        M = self.cfg.env.M_queries
        if synthesize:
            from ..grammar.query_synthesizer import synthesize_queries
            synth = synthesize_queries(
                grammar, n=M, max_depth=3, max_len=6,
                rng=self._rng, existing=queries_raw,
            )
            selected = synth[:M]
            if len(selected) < M:
                extra = queries_raw[:M - len(selected)]
                selected.extend(extra)
        else:
            if len(queries_raw) >= M:
                indices = self._rng.choice(len(queries_raw), size=M, replace=False)
                selected = [queries_raw[i] for i in sorted(indices)]
            else:
                # Recycle queries when pool is too small
                # Each re-use gets different risk/options via V2 pool
                repeats = (M // len(queries_raw)) + 1
                expanded = (queries_raw * repeats)[:M]
                self._rng.shuffle(expanded)
                selected = list(expanded)

        # Build QueryState for each
        query_states = []
        for qi, ex in enumerate(selected):
            menu, quota_info = self._generate_v2_menu(ex)
            qs = QueryState(
                query_id=qi,
                target_output=list(ex.output),
                true_program=list(ex.words),
                hp=self.cfg.env.H_0,
                max_rounds=self.cfg.env.T_max,
                max_refreshes=self.cfg.env.max_refreshes,
                enforce_max_refreshes=getattr(self.cfg.env, "enforce_max_refreshes", False),
                menu=menu,
            )
            # Phase 6E: attach diagnostic sidecar labels
            if quota_info is not None:
                qs.option_confound_types = quota_info.get("confound_types", {})
                qs.option_diag_labels = quota_info.get("diag_labels", {})
                qs._quota_info = quota_info  # Phase 6H.6: strict diagnostics
            query_states.append(qs)

        teach_budget = getattr(self.cfg.env, 'teach_step_budget', 0)

        if teach_budget > 0:
            # Budget mode: pre-allocate enough teach queries (worst case: 1 step each)
            max_teach_qs = min(teach_budget, max(0, len(query_states) - self.cfg.env.N_obs))
            block = BlockState(
                block_id=block_id,
                support_examples=support,
                queries=query_states,
                obs_phase_queries=min(self.cfg.env.N_obs, len(query_states)),
                teach_phase_queries=max_teach_qs,
                eval_phase_queries=min(
                    self.cfg.env.N_eval,
                    max(0, len(query_states) - self.cfg.env.N_obs - max_teach_qs)),
                teach_step_budget=teach_budget,
                teach_steps_used=0,
            )
        else:
            block = BlockState(
                block_id=block_id,
                support_examples=support,
                queries=query_states,
                obs_phase_queries=min(self.cfg.env.N_obs, len(query_states)),
                teach_phase_queries=min(
                    self.cfg.env.N_teach,
                    max(0, len(query_states) - self.cfg.env.N_obs)),
                eval_phase_queries=min(
                    self.cfg.env.N_eval,
                    max(0, len(query_states) - self.cfg.env.N_obs - self.cfg.env.N_teach)),
            )
        return block

    def _generate_v2_menu(self, ex: Example):
        """Generate a V2 menu: diverse valid-only options + risk.

        Uses ProgramPool for valid distractors with cell-level diversity.
        Falls back to V1 generator if pool is unavailable.

        Returns:
            (menu, quota_info_or_None)
        """
        gen_mode = getattr(self.cfg.env, 'generator_mode', 'v2_overlap')

        # Phase 6E: diagnostic quota generator (includes ablation modes)
        if gen_mode.startswith('diagnostic_quota') and self._pool is not None:
            strict = getattr(self.cfg.env, 'diagnostic_quota_strict', False)
            max_attempts = 5 if strict else 1

            best_menu, best_qi = None, None
            for attempt in range(max_attempts):
                # Use offset rng for retries to get different candidates
                attempt_rng = (self._rng if attempt == 0
                               else np.random.default_rng(
                                   self._rng.integers(0, 2**31) + attempt))
                menu, quota_info = generate_menu_diagnostic(
                    target_output=ex.output,
                    true_program=ex.words,
                    pool=self._pool,
                    danger_model=self._danger_model,
                    K=self.cfg.env.K,
                    m=self.cfg.env.danger_dim,
                    rng=attempt_rng,
                    quota_mode=gen_mode,
                )

                # For fallback path: re-assign risk classes from env config
                if quota_info.get('fallback_used', False):
                    K = len(menu)
                    n_safe = max(0, K - self.cfg.env.n_risky)
                    risk_classes = self._danger_model.assign_risk_classes(
                        K, n_safe, attempt_rng)
                    for i, opt in enumerate(menu):
                        rc = risk_classes[i]
                        opt.risk_class = rc
                        opt.danger_vec = self._danger_model.sample_danger_vec(
                            rc, attempt_rng)
                    # Re-label with correct risk classes
                    from ..grammar.option_generator_diagnostic import (
                        _label_menu_post_hoc,
                    )
                    quota_info["confound_types"] = {}
                    quota_info["diag_labels"] = {}
                    _label_menu_post_hoc(menu, tuple(ex.output), quota_info)

                # Check quota from FINAL post-hoc labels
                final_labels = quota_info.get("diag_labels", {})
                final_has_safe = any(
                    v == "safe_diagnostic_wrong" for v in final_labels.values())
                final_has_bounded = any(
                    v == "bounded_diagnostic_wrong" for v in final_labels.values())
                final_has_lure = any(
                    v == "high_risk_lure" for v in final_labels.values())
                final_quota_met = final_has_safe and final_has_bounded and final_has_lure

                # Update quota_info with strict diagnostics
                quota_info["quota_met"] = final_quota_met
                quota_info["strict_attempt_count"] = attempt + 1
                quota_info["strict_quota_satisfied"] = final_quota_met
                quota_info["strict_mode"] = strict

                if final_quota_met or not strict:
                    best_menu, best_qi = menu, quota_info
                    break
                # Keep last attempt as fallback
                best_menu, best_qi = menu, quota_info

            # If strict exhausted all attempts without satisfaction
            if strict and not best_qi.get("strict_quota_satisfied", False):
                best_qi["strict_fallback"] = True
            else:
                best_qi["strict_fallback"] = False

            return best_menu, best_qi

        # Standard v2_overlap path
        if self._pool is not None:
            base_menu = generate_menu_v2(
                target_output=ex.output,
                true_program=ex.words,
                pool=self._pool,
                danger_model=self._danger_model,
                K=self.cfg.env.K,
                m=self.cfg.env.danger_dim,
                rng=self._rng,
            )
        else:
            # Fallback to V1
            base_menu = generate_menu(
                target_output=ex.output,
                true_program=ex.words,
                grammar=self._grammar,
                support=self._support,
                danger_model=self._danger_model,
                K=self.cfg.env.K,
                m=self.cfg.env.danger_dim,
                rng=self._rng,
            )

        # Assign V2 risk classes — canonical: derive n_safe from n_risky
        K = len(base_menu)
        n_safe = max(0, K - self.cfg.env.n_risky)
        risk_classes = self._danger_model.assign_risk_classes(
            K, n_safe, self._rng)

        for i, opt in enumerate(base_menu):
            rc = risk_classes[i]
            opt.risk_class = rc
            opt.danger_vec = self._danger_model.sample_danger_vec(rc, self._rng)

        # Always label for metrics (even in v2_overlap mode)
        from ..grammar.option_generator_diagnostic import _label_menu_post_hoc
        quota_info = {"quota_mode": "v2_overlap", "quota_met": False,
                      "fallback_used": False, "confound_types": {},
                      "diag_labels": {}}
        _label_menu_post_hoc(base_menu, tuple(ex.output), quota_info)

        return base_menu, quota_info

    def _resample_risk(self, qs: QueryState) -> None:
        """V2 risk-only refresh: keep text, re-sample risk classes + vectors."""
        K = len(qs.menu)
        n_safe = max(0, K - self.cfg.env.n_risky)
        risk_classes = self._danger_model.assign_risk_classes(
            K, n_safe, self._rng)

        for i, opt in enumerate(qs.menu):
            rc = risk_classes[i]
            opt.risk_class = rc
            opt.danger_vec = self._danger_model.sample_danger_vec(rc, self._rng)

    # ── Tutor action ────────────────────────────────────────────

    def tutor_act(
        self,
        block: BlockState,
        action: str,
        ban_index: Optional[int] = None,
        hint_index: Optional[int] = None,
        highlight_cells: Optional[Tuple[int, ...]] = None,
        shortlist_indices: Optional[list] = None,
    ) -> TutorStep:
        """Execute a tutor action on the current query."""
        qs = block.current_query
        if qs is None or qs.done:
            raise ValueError("No active query")

        round_t = qs.rounds_used

        # Observation phase OR evaluation phase: tutor can only WAIT
        if (block.in_observation_phase or block.in_evaluation_phase) and action != "WAIT":
            action = "WAIT"

        if action == "WAIT":
            step = apply_wait(qs, round_t)
        elif action == "RISK_HINT":
            if hint_index is None:
                raise ValueError("RISK_HINT requires hint_index")
            step = apply_risk_hint(qs, hint_index, round_t,
                                   eta=self.cfg.learner.eta_hint)
        elif action == "BAN":
            if ban_index is None:
                raise ValueError("BAN requires ban_index")
            step = apply_ban(qs, ban_index, round_t)
        elif action == "HIGHLIGHT":
            if highlight_cells is None:
                raise ValueError("HIGHLIGHT requires highlight_cells")
            step = apply_highlight(
                qs, highlight_cells,
                max_cells=self.cfg.tutor.max_highlight_cells,
                round_t=round_t)
        elif action == "SHORTLIST":
            if shortlist_indices is None:
                raise ValueError("SHORTLIST requires shortlist_indices")
            step = apply_shortlist(qs, shortlist_indices, round_t)
        elif action == "MIX":
            if ban_index is None:
                raise ValueError("MIX requires ban_index")
            if highlight_cells is None:
                raise ValueError("MIX requires highlight_cells")
            step = apply_mix(
                qs, ban_index, highlight_cells,
                round_t=round_t,
                max_cells=self.cfg.tutor.max_highlight_cells,
            )
        elif action == "SKIP":
            step = apply_skip(qs, round_t)
            block.total_skips += 1
        elif action == "PASS":
            from ..env.interventions import apply_pass
            step = apply_pass(qs, round_t)
            block.total_skips += 1  # count as skip for metrics
        else:
            raise ValueError(f"Unknown tutor action: {action}")

        block.tutor_trace.append(step)
        return step

    # ── Learner action ──────────────────────────────────────────

    def learner_act(
        self,
        block: BlockState,
        action: str,
        pick_index: Optional[int] = None,
    ) -> LearnerStep:
        """Execute a learner action on the current query."""
        qs = block.current_query
        if qs is None or qs.done:
            raise ValueError("No active query")

        round_t = qs.rounds_used
        hp_before = qs.hp

        if action == "refresh":
            return self._do_refresh(block, qs, round_t, hp_before)
        elif action == "pick":
            if pick_index is None:
                raise ValueError("pick requires pick_index")
            return self._do_pick(block, qs, pick_index, round_t, hp_before)
        else:
            raise ValueError(f"Unknown learner action: {action}")

    def _do_refresh(
        self,
        block: BlockState,
        qs: QueryState,
        round_t: int,
        hp_before: int,
    ) -> LearnerStep:
        """V2 REFRESH: keep text, re-sample risk only.

        Legacy mode keeps refresh unlimited. Controlled mode enforces the
        per-query refresh cap before spending the round.
        """
        if (
            qs.enforce_max_refreshes
            and qs.refreshes_used >= qs.max_refreshes
        ):
            raise ValueError(
                f"Refresh cap exceeded for query {qs.query_id}: "
                f"{qs.refreshes_used} >= {qs.max_refreshes}"
            )

        qs.refreshes_used += 1
        block.total_refreshes += 1

        self._resample_risk(qs)
        # V2: clear BAN/RISK_HINT only, preserve HIGHLIGHT
        clear_menu_interventions(qs)

        qs.rounds_used += 1
        step = LearnerStep(
            round_t=round_t, query_id=qs.query_id,
            action="refresh", hp_before=hp_before,
            hp_after=hp_before, menu_size=len(qs.menu),
        )
        block.learner_trace.append(step)
        block.total_rounds += 1
        # Budget tracking: count each step during teach phase
        if block.in_teaching_phase:
            block.teach_steps_used += 1
        self._check_query_end(block, qs)
        return step

    def _do_pick(
        self,
        block: BlockState,
        qs: QueryState,
        pick_index: int,
        round_t: int,
        hp_before: int,
    ) -> LearnerStep:
        """Handle PICK: check correctness, V2 discrete damage, update HP."""
        active = get_active_menu(qs)
        option = None
        for o in qs.menu:
            if o.index == pick_index:
                option = o
                break
        if option is None:
            raise ValueError(f"Invalid pick_index {pick_index}")
        if pick_index in qs.banned_indices:
            raise ValueError(f"Option {pick_index} is banned")
        # Enforce shortlist constraint: if a shortlist is active, learner must
        # pick from it. This is the env-level enforcement of the invariant
        # final_choice ∈ S.
        if qs.shortlisted_indices is not None and pick_index not in qs.shortlisted_indices:
            raise ValueError(
                f"Option {pick_index} not in active shortlist {qs.shortlisted_indices}"
            )

        correct = option.is_correct
        damage = 0

        if correct:
            qs.success = True
            qs.done = True
            block.total_correct += 1
            # Phase 6H: clear post-reveal phase on success
            qs.post_reveal_phase = False
        else:
            # V2: damage = risk_class (deterministic)
            rendered = option.rendered_output
            if rendered is None:
                rendered = TaskAdapter.render(option.text, self._grammar) or []

            damage = option.risk_class  # V2: discrete deterministic

            qs.hp = max(0, qs.hp - damage)
            block.total_damage += damage

            reveal = RevealEvent(
                round_t=round_t,
                option_index=pick_index,
                option_text=list(option.text),
                revealed_output=rendered,
                damage=damage,
                expected_damage=float(option.risk_class),
                danger_vec=option.danger_vec.copy(),
                risk_class=option.risk_class,
            )
            qs.reveal_history.append(reveal)

            # Phase 6H: update query trajectory state after wrong pick
            self._update_query_trajectory_after_wrong_pick(qs, option, round_t)

        qs.rounds_used += 1
        block.total_rounds += 1

        step = LearnerStep(
            round_t=round_t, query_id=qs.query_id,
            action="pick", pick_index=pick_index,
            correct=correct, damage=damage,
            hp_before=hp_before, hp_after=qs.hp,
            menu_size=len(active),
        )
        block.learner_trace.append(step)
        # Budget tracking: count each step during teach phase
        if block.in_teaching_phase:
            block.teach_steps_used += 1
        self._check_query_end(block, qs)
        return step

    # ── Phase 6H: trajectory state update ────────────────────────

    def _update_query_trajectory_after_wrong_pick(
        self, qs: QueryState, option: Option, round_t: int
    ) -> None:
        """Update query-level trajectory state after a wrong pick.

        Uses sidecar diagnostic labels to track reveal type and set
        post_reveal_phase for CONSOLIDATE logic in sparse tutor.
        """
        label = qs.option_diag_labels.get(option.index, "")

        if label == "safe_diagnostic_wrong":
            qs.n_safe_diag_wrong_reveals += 1
        elif label == "bounded_diagnostic_wrong":
            qs.n_bounded_diag_wrong_reveals += 1
        elif label == "high_risk_lure":
            qs.n_high_risk_wrong_reveals += 1

        qs.last_wrong_diag_label = label
        qs.last_reveal_round_t = round_t
        qs.last_reveal_option_index = option.index

        # Enter post-reveal phase after any diagnostic reveal
        if label in ("safe_diagnostic_wrong", "bounded_diagnostic_wrong"):
            qs.post_reveal_phase = True

    # ── Query termination ───────────────────────────────────────

    def _check_query_end(self, block: BlockState, qs: QueryState) -> None:
        """Check if current query should end, advance if so."""
        def _record_grace_loss(reason: str) -> None:
            if not getattr(qs, "after_highlight_grace_round", False):
                return
            gm = getattr(block, "_grace_metrics", None)
            if gm is None:
                block._grace_metrics = {
                    "set": 0,
                    "eligible_next_round": 0,
                    "next_tutor_called": 0,
                    "count": 0,
                    "chosen_wait": 0,
                    "chosen_override": 0,
                    "consumed": 0,
                    "override": 0,
                    "blocked_by_protect": 0,
                    "blocked_by_deadline": 0,
                    "did_not_reach_tutor_decision": 0,
                    "lost_query_succeeded": 0,
                    "lost_wrong_terminal": 0,
                    "lost_max_round": 0,
                    "flag_reset_without_consumption": 0,
                }
                gm = block._grace_metrics
            gm["did_not_reach_tutor_decision"] = gm.get("did_not_reach_tutor_decision", 0) + 1
            gm[reason] = gm.get(reason, 0) + 1
            qs.after_highlight_grace_round = False

        if qs.done:
            if qs.success:
                _record_grace_loss("lost_query_succeeded")
            self._advance_query(block)
            return
        if qs.hp <= 0:
            _record_grace_loss("lost_wrong_terminal")
            qs.done = True
            self._advance_query(block)
            return
        if qs.rounds_used >= qs.max_rounds:
            _record_grace_loss("lost_max_round")
            qs.done = True
            self._advance_query(block)
            return

    def _advance_query(self, block: BlockState) -> None:
        """Move to the next query in the block."""
        block.current_query_idx += 1
        if block.current_query_idx >= len(block.queries):
            block.done = True

    # ── Convenience: combined step ──────────────────────────────

    def step_round(
        self,
        block: BlockState,
        tutor_action: str = "WAIT",
        tutor_kwargs: Optional[dict] = None,
        learner_action: str = "pick",
        learner_kwargs: Optional[dict] = None,
    ) -> Tuple[Optional[TutorStep], Optional[LearnerStep]]:
        """Execute one full round: tutor + learner."""
        if block.done:
            return None, None

        qs = block.current_query
        if qs is None or qs.done:
            return None, None

        tutor_kwargs = tutor_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        ts = self.tutor_act(block, tutor_action, **tutor_kwargs)

        if qs.done:
            return ts, None

        ls = self.learner_act(block, learner_action, **learner_kwargs)
        return ts, ls

    def force_learner_pick(
        self,
        block: BlockState,
        pick_index: int,
    ) -> LearnerStep:
        """Force a specific learner pick (for scripted mechanism probes).

        Bypasses the learner's decision logic.  The pick is executed
        through the normal env pick transition (damage, reveal, HP update).
        Used only in scripted_protocols.py.
        """
        return self.learner_act(block, "pick", pick_index=pick_index)

    # ── Metrics ─────────────────────────────────────────────────

    @staticmethod
    def get_block_metrics(block: BlockState) -> dict:
        """Compute aggregate block metrics."""
        n_queries = len(block.queries)
        return {
            "n_queries": n_queries,
            "total_correct": block.total_correct,
            "solve_rate": block.total_correct / max(n_queries, 1),
            "total_damage": block.total_damage,
            "avg_damage": block.total_damage / max(n_queries, 1),
            "total_rounds": block.total_rounds,
            "avg_rounds": block.total_rounds / max(n_queries, 1),
            "total_skips": block.total_skips,
            "total_refreshes": block.total_refreshes,
            "observation_phase": block.obs_phase_queries,
            "teaching_phase": block.teach_phase_queries,
        }

    @property
    def danger_model(self) -> Optional[DangerModel]:
        return self._danger_model
