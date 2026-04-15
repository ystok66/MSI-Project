"""
Phase 3 diagnostic: investigate divergence = 0, query source, and parameter values.
Answers:
  Q1: Where do queries come from? Generate 10 examples for inspection.
  Q2: What are actual n_sup, n_obs, n_teach, n_eval values?
  Q3: At which step does divergence become 0?
  Q4: Is divergence=0 a bug? — Deep comparison of shadow vs real state.
"""
import sys, os
import numpy as np
import copy

proj_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, '..', 'BASIC'))

from cls_color_selection.config import FullConfig
from cls_color_selection.environment.grammar_task_env import GrammarTaskEnv
from cls_color_selection.learner.cls_wrapper import CLSSequencePredictor
from cls_color_selection.learner.target_predictor import TargetPredictor
from cls_color_selection.learner.risk_belief import DangerTypeBelief
from cls_color_selection.learner.feedback_update import FeedbackUpdater
from cls_color_selection.learner.policy import ColorSelectionPolicy
from cls_color_selection.learner.memory import QueryMemory
from cls_color_selection.tutor_api.tutor_shadow import ShadowTutor
from cls_color_selection.tutor_api.shadow_clone import create_shadow_snapshot
from cls_color_selection.tutor_api.shadow_update import shadow_predict_target, _reconstruct_shadow_cls
from cls_color_selection.tutor_api.joint_debug import (
    measure_grammar_divergence, measure_risk_divergence, compute_full_divergence,
)
from cls_color_selection.tutor_api.tutor_state import TutorBelief
from cls_color_selection.tutor_api.observation import run_observation_phase
from cls_color_selection.tutor_api.belief_update import initialize_belief_from_observation
from cls_color_selection.tutor_api.dummy_tutor import NoTutor
from cls_color_selection.experiments.run_phase2 import run_query_with_tutor

lines = []

def log(s):
    print(s)
    lines.append(s)

data_dir = os.path.normpath(os.path.join(proj_root, '..', 'BASIC', 'cls_learner', 'data'))

# ────────────────────────────────────────────────────────────────
# Q1: Query source
# ────────────────────────────────────────────────────────────────
log("=" * 60)
log("# Q1: Query Source")
log("=" * 60)

cfg = FullConfig()
rng = np.random.default_rng(42)
env = GrammarTaskEnv(cfg, rng)
task_path = os.path.join(data_dir, '000001.txt')
support, queries, grammar = env.load_task(task_path)

log(f"\nTask file: {task_path}")
log(f"Support count: {len(support)}")
log(f"Query count: {len(queries)}")
log(f"Grammar nouns: {grammar.nouns}")
log(f"Grammar colors: {grammar.colors}")
log(f"Grammar rules: {grammar.rules}")

log(f"\n## All Queries from 000001.txt:")
for i, q in enumerate(queries):
    log(f"  Q{i}: IN: {' '.join(q.words)}  OUT: {' '.join(q.output)}")
    log(f"       output length = {len(q.output)}")

# Check more files
log(f"\n## Query counts per task file:")
for tid in range(1, 6):
    tp = os.path.join(data_dir, f'{tid:06d}.txt')
    s, q, g = env.load_task(tp)
    log(f"  {tid:06d}.txt: {len(s)} support, {len(q)} query, colors={g.colors}")

# ────────────────────────────────────────────────────────────────
# Q2: Actual parameter values
# ────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("# Q2: Parameter Values Used in Experiments")
log("=" * 60)

cfg = FullConfig()
log(f"\n## Default Config:")
log(f"  n_sup = {cfg.learner.n_sup}")
log(f"  n_em = {cfg.learner.n_em}")
log(f"  n_obs_queries = {cfg.exp.n_obs_queries}")
log(f"  n_teach_queries = {cfg.exp.n_teach_queries}")
log(f"  n_eval_queries = {cfg.exp.n_eval_queries}")
log(f"  use_observation_phase = {cfg.tutor.use_observation_phase}")

# Show the actual split on task 000001
env2 = GrammarTaskEnv(cfg, np.random.default_rng(42))
_, queries2, _ = env2.load_task(task_path)
available = len(queries2)
n_obs = cfg.exp.n_obs_queries if cfg.tutor.use_observation_phase else 0
n_teach = cfg.exp.n_teach_queries
n_eval = cfg.exp.n_eval_queries
total_needed = n_obs + n_teach + n_eval

log(f"\n## Query Split for 000001.txt (available={available}):")
log(f"  Requested: n_obs={n_obs}, n_teach={n_teach}, n_eval={n_eval}")
log(f"  Total needed: {total_needed}")
log(f"  OVERFLOW: {'YES' if total_needed > available else 'NO'} ({total_needed} > {available})")

if total_needed > available:
    scale = available / total_needed
    n_obs_actual = int(n_obs * scale)
    n_teach_actual = int(n_teach * scale)
    n_eval_actual = available - n_obs_actual - n_teach_actual
    log(f"  Scaled: n_obs={n_obs_actual}, n_teach={n_teach_actual}, n_eval={n_eval_actual}")
    log(f"  PROBLEM: Only {n_teach_actual} teach queries and {n_eval_actual} eval queries!")
else:
    n_obs_actual = n_obs
    n_teach_actual = n_teach
    n_eval_actual = n_eval

log(f"\n## Obs queries:")
for i in range(n_obs_actual):
    q = queries2[i]
    log(f"  obs[{i}]: {' '.join(q.words)} -> {' '.join(q.output)}")
log(f"\n## Teach queries:")
for i in range(n_obs_actual, n_obs_actual + n_teach_actual):
    q = queries2[i]
    log(f"  teach[{i-n_obs_actual}]: {' '.join(q.words)} -> {' '.join(q.output)}")
log(f"\n## Eval queries:")
for i in range(n_obs_actual + n_teach_actual, n_obs_actual + n_teach_actual + n_eval_actual):
    q = queries2[i]
    log(f"  eval[{i-n_obs_actual-n_teach_actual}]: {' '.join(q.words)} -> {' '.join(q.output)}")

# ────────────────────────────────────────────────────────────────
# Q3 & Q4: Deep divergence investigation
# ────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("# Q3 & Q4: Divergence Investigation")
log("=" * 60)

cfg3 = FullConfig()
rng3 = np.random.default_rng(42)
env3 = GrammarTaskEnv(cfg3, rng3)
support3, queries3, grammar3 = env3.load_task(task_path)

predictor = CLSSequencePredictor(cfg3.learner)
sub_support = support3[:cfg3.learner.n_sup]
predictor.fit_support(sub_support)
target_pred = TargetPredictor(predictor)

risk_belief = DangerTypeBelief(
    n_danger_types=cfg3.env.n_danger_types,
    danger_dim=cfg3.env.danger_dim,
    obs_sigma=cfg3.env.obs_sigma,
    prior_safe=cfg3.learner.risk_prior_safe,
)
risk_belief.set_prototypes(
    env3.danger_model.prototypes,
    np.ones_like(env3.danger_model.prototypes) * cfg3.env.cluster_sigma**2,
)

# Create shadow snapshot
shadow = create_shadow_snapshot(
    predictor, risk_belief, cfg3.learner, sub_support, fidelity='exact',
)

log(f"\n## Initial shadow state:")
log(f"  Shadow grammar words: {sorted(shadow.grammar.keys())}")
log(f"  Shadow fidelity: {shadow.fidelity}")

# Compare initial prediction on EVERY query
log(f"\n## Initial prediction comparison (shadow vs real) on ALL queries:")
all_agree = True
for i, q in enumerate(queries3):
    real_pred = predictor.predict_target(q.words)
    shadow_pred = shadow_predict_target(shadow, q.words)
    agree = real_pred == shadow_pred
    if not agree:
        all_agree = False
    log(f"  Q{i} ({' '.join(q.words)}):")
    log(f"    real:   {real_pred}")
    log(f"    shadow: {shadow_pred}")
    log(f"    agree:  {agree}")

log(f"\n  ALL AGREE: {all_agree}")

# KEY: check if shadow is using the SAME CLSAgent via shared reference
log(f"\n## Object identity check (BUG HUNT):")
real_agent = predictor.get_agent()
shadow_agent, shadow_predictor = _reconstruct_shadow_cls(shadow)

if shadow_agent is real_agent:
    log(f"  *** BUG: shadow_agent IS real_agent (shared reference!) ***")
else:
    log(f"  OK: shadow_agent is NOT real_agent (independent)")

if shadow_agent is not None:
    real_lib = real_agent.cortex.library
    shadow_lib = shadow_agent.cortex.library
    log(f"  Real library id: {id(real_lib)}")
    log(f"  Shadow library id: {id(shadow_lib)}")
    log(f"  Same library: {real_lib is shadow_lib}")

    # Compare actual concept counts
    for word in sorted(real_lib.keys())[:4]:
        rc = real_lib[word]
        sc = shadow_lib[word]
        log(f"\n  Word '{word}':")
        log(f"    Real role_counts:   {dict(rc.role_counts)}")
        log(f"    Shadow role_counts: {dict(sc.role_counts)}")
        real_emit_w = rc.emit_stats.get('sum_w', 0)
        shadow_emit_w = sc.emit_stats.get('sum_w', 0)
        log(f"    Real emit_w:   {real_emit_w:.4f}")
        log(f"    Shadow emit_w: {shadow_emit_w:.4f}")
        match = (dict(rc.role_counts) == dict(sc.role_counts))
        log(f"    Role counts match: {match}")

# Now simulate ONE teaching round and check divergence AFTER
log(f"\n## After teaching one query with feedback:")

policy = ColorSelectionPolicy(cfg3.learner)
feedback_updater = FeedbackUpdater(cfg3.learner)

# Deep copy so we can compare
real_risk_before = copy.deepcopy(risk_belief)

# Run one teach query with real learner
teach_q = queries3[n_obs_actual]  # first teach query
y_star = target_pred.predict_target(teach_q.words)
state = env3.init_query(teach_q, query_id=0, target_output=y_star)
memory = QueryMemory()

# Use NoTutor (no intervention) so feedback modifies grammar
tutor = NoTutor()
result, diag = run_query_with_tutor(
    env3, state, policy, risk_belief, feedback_updater,
    predictor, target_pred, tutor, memory, rng3, cfg3,
    belief=None, immortal=False, enable_feedback=True,
)
log(f"  Teach query: {' '.join(teach_q.words)}")
log(f"  Outcome: {result.outcome.name}")
log(f"  Confirms: {result.confirm_count}")

# Check: did feedback modify the real library?
target_pred.invalidate_all()
real_pred_after = predictor.predict_target(teach_q.words)
shadow_pred_after = shadow_predict_target(shadow, teach_q.words)

log(f"\n  After teaching, predictions:")
log(f"    real:   {real_pred_after}")
log(f"    shadow: {shadow_pred_after}")
log(f"    agree:  {real_pred_after == shadow_pred_after}")

if result.confirm_count > 0 and result.outcome.name == 'WRONG':
    log(f"\n  *** Feedback was applied (confirm_count > 0 with wrong outcome)")
    log(f"  *** Shadow should now DIVERGE from real if update happened")
else:
    log(f"\n  No feedback update occurred (outcome={result.outcome.name})")

# Check library changes
for word in sorted(predictor.get_library().keys())[:3]:
    rc_now = predictor.get_library()[word]
    sc_now = shadow.grammar.get(word)
    if sc_now:
        match = all(
            abs(rc_now.role_counts.get(r, 0) - sc_now.role_counts.get(r, 0)) < 1e-10
            for r in rc_now.role_counts
        )
        if not match:
            log(f"  '{word}': DIVERGED after teaching")
            log(f"    real:   {dict(rc_now.role_counts)}")
            log(f"    shadow: {sc_now.role_counts}")
        else:
            log(f"  '{word}': still matches")

# ────────────────────────────────────────────────────────────────
# Final diagnosis
# ────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("# DIAGNOSIS SUMMARY")
log("=" * 60)

log(f"""
## Key Facts:

1. Queries come from: txt files directly (parse_task_file)
   - NOT generated, just parsed from BASIC/cls_learner/data/*.txt
   - Task 000001 has {len(queries)} queries total

2. Default config: n_obs={cfg.exp.n_obs_queries}, n_teach={cfg.exp.n_teach_queries}, n_eval={cfg.exp.n_eval_queries}
   - Total needed: {cfg.exp.n_obs_queries + cfg.exp.n_teach_queries + cfg.exp.n_eval_queries}
   - Available in 000001.txt: {len(queries)}
   - OVERFLOW: {cfg.exp.n_obs_queries + cfg.exp.n_teach_queries + cfg.exp.n_eval_queries > len(queries)}

3. Shadow divergence = 0 root cause analysis:
   The shadow is reconstructed by:
     a. Creating a new CLSAgent
     b. Calling agent.study(support) to establish vocabulary + structure
     c. Overwriting library counts from snapshot
   Since the support set is IDENTICAL and updates are deterministic,
   the shadow's grammar state IS the real learner's grammar state
   (at the time of snapshot creation).
   
   HOWEVER: during teaching, the REAL learner gets feedback updates
   (differential M-step modifying role_counts/emit_stats), but the
   SHADOW does NOT get updated in sync. The divergence measurement
   only happens BEFORE each teach query, and the shadow is initialized
   just before teaching starts. If no feedback has been applied yet,
   divergence is naturally 0.
   
   The BUG: divergence is measured at the START of each teach query
   but the shadow state is NEVER updated during teaching loops in
   run_phase3.py. The shadow only updates during ShadowTutor.on_select()
   (risk updates) and on_confirm_fail() (hint decisions), but the
   grammar side is not sync'd. Meanwhile the divergence test only
   measures the shadow's INITIAL state.
""")

# Write to file
out_dir = os.path.join(proj_root, 'results', 'phase3_diagnostic')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'diagnostic_report.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nFull report: {out_path}")
