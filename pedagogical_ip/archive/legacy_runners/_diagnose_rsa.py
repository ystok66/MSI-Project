"""Diagnose why RSA M_true goes down instead of up."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.rsa_warning_channel import (
    RSAWarningChannel, RSABeliefState, RSAUtterance, ALL_HYPOTHESES,
    literal_listener, pragmatic_speaker, N_HYPOTHESES,
)
from src.agents.warning_update import (
    map_segment_to_rsa_context, map_legacy_to_rsa_utterance, Utterance
)

# Run one episode and inspect the diagnostics
runner = LatticeV2Runner()
s = runner.reset(seed=42, latent_mode=True, warning_mode='fixed',
                 scenario_family='fork_trap', warning_variant='rsa_obs_s1')
while not s.done:
    s = runner.step(s)

# Print detailed diagnostics
print("=== RSA Warning Diagnostics ===")
for i, d in enumerate(s.rsa_warn_diagnostics):
    print(f"\nWarning #{i}")
    for k, v in d.items():
        if isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")

# Check the segment topology
print("\n=== Segment Topology ===")
for seg in s.meta.segments:
    ctx = map_segment_to_rsa_context(seg)
    print(f"  Segment {seg.index}: risky_row={seg.risky_row}, safe_row={seg.safe_row}")
    print(f"  risky_side (from ctx): {ctx['risky_side']}")
    print(f"  risky_cells: {seg.risky_cells[:3]}")
    print(f"  safe_cells: {seg.safe_cells[:3]}")
    print()

# Manually trace RSA inference
print("\n=== Manual RSA Trace ===")
seg = s.meta.segments[0]
ctx = map_segment_to_rsa_context(seg)
risky_side = ctx['risky_side']
print(f"risky_side = {risky_side}")

# What utterance gets selected?
utt = Utterance.RISKY_TEXTURE_AHEAD  # fixed warning mode
rsa_utt = map_legacy_to_rsa_utterance(utt, risky_side)
print(f"legacy_utt = {utt.value}, mapped to rsa_utt = {rsa_utt.value}")

# What does L0 give?
prior = np.ones(N_HYPOTHESES) / N_HYPOTHESES
l0_post = literal_listener(rsa_utt, ctx, prior, lambda_sem=3.0)
print(f"\nL0 posterior: [{', '.join(f'{x:.4f}' for x in l0_post)}]")
for h, p in zip(ALL_HYPOTHESES, l0_post):
    print(f"  {h.value}: {p:.4f}")

# What does S1 give?
from src.agents.rsa_warning_channel import pragmatic_speaker
ch = RSAWarningChannel()
bs = RSABeliefState()
info = ch.update_belief(bs, rsa_utt, ctx, variant="s1")
print(f"\nS1 posterior: [{', '.join(f'{x:.4f}' for x in bs.belief)}]")
for h, p in zip(ALL_HYPOTHESES, bs.belief):
    print(f"  {h.value}: {p:.4f}")

# What is the TRUE hypothesis?
if seg.risky_row <= 2:
    true_hyp = "left_risky"
else:
    true_hyp = "right_risky"
print(f"\nTrue hypothesis: {true_hyp}")
print(f"True hypothesis in indices: left=0, right=1")
true_idx = 0 if true_hyp == "left_risky" else 1
print(f"M_true before = {prior[true_idx]:.4f}")
print(f"M_true after  = {bs.belief[true_idx]:.4f}")
