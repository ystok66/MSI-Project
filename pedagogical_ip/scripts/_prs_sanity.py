"""Quick PRS session sanity check."""
import sys
sys.path.insert(0, ".")

from src.envs.prs_session import PRSSession, SessionConfig

cfg = SessionConfig(
    session_seed=42,
    curriculum="mixed",
    tutor_strategy="selective",
    block_a_size=3,
    block_b_size=2,
    block_c_size=2,
    block_d_size=2,
    difficulty="hard",
    persist_agent_memory=True,
)
s = PRSSession(cfg)
r = s.run_session()

print("=== SESSION SANITY ===")
for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D",
          "surv_A", "surv_B", "surv_C", "surv_D",
          "transfer_gap_C", "transfer_gap_D",
          "dependence_proxy", "n_A", "n_B", "n_C", "n_D"]:
    print(f"  {k:25s} = {r.get(k, 'N/A')}")

# Check block results
for bid in ["A", "B", "C", "D"]:
    recs = r["block_results"][bid]
    families = [rec["family"].split("_")[0] for rec in recs]
    shifts = [rec.get("shift", "none") for rec in recs]
    tutors = [rec["tutor_enabled"] for rec in recs]
    errors = [rec.get("error", "") for rec in recs]
    updates = [rec.get("predictor_n_updates", 0) for rec in recs]
    print(f"  Block {bid}: n={len(recs)} fam={families} shift={shifts} "
          f"tutor={tutors} updates={updates}")
    if any(errors):
        print(f"    ERRORS: {errors}")

print("\nSESSION SANITY OK")
