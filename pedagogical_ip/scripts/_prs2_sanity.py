"""Quick sanity: session_shared WorldWeights override works."""
import sys; sys.path.insert(0, ".")
import numpy as np

from src.envs.prs_session import PRSSession, SessionConfig

# Test 1: session_shared — all episodes use same WorldWeights
cfg = SessionConfig(
    session_seed=42, curriculum="gtet_only", tutor_strategy="selective",
    block_a_size=3, block_b_size=2, block_c_size=2, block_d_size=2,
    difficulty="medium", persist_agent_memory=True,
    weight_mode="session_shared",
)
s = PRSSession(cfg)
r = s.run_session()

print("=== PRS-2 SANITY: session_shared ===")
for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D", "n_A", "n_B", "n_C", "n_D"]:
    print(f"  {k:25s} = {r.get(k, 'N/A')}")

for bid in ["A", "B", "C", "D"]:
    recs = r["block_results"][bid]
    updates = [rec.get("predictor_n_updates", 0) for rec in recs]
    wm = [rec.get("weight_mode", "?") for rec in recs]
    print(f"  Block {bid}: updates={updates} weight_mode={wm}")

# Test 2: episode_random — verify negative control
cfg2 = SessionConfig(
    session_seed=42, curriculum="gtet_only", tutor_strategy="selective",
    block_a_size=3, block_b_size=2, block_c_size=2, block_d_size=2,
    difficulty="medium", persist_agent_memory=True,
    weight_mode="episode_random",
)
s2 = PRSSession(cfg2)
r2 = s2.run_session()

print("\n=== PRS-2 SANITY: episode_random ===")
for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D"]:
    print(f"  {k:25s} = {r2.get(k, 'N/A')}")

print("\n=== COMPARISON ===")
for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D"]:
    v1 = r.get(k, 0)
    v2 = r2.get(k, 0)
    print(f"  {k}: shared={v1:.3f} random={v2:.3f} Δ={v1-v2:+.3f}")

print("\nPRS-2 SANITY OK")
