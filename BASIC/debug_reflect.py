"""Debug reflect kappa merging - ASCII safe output."""
import sys, os, numpy as np

from rsa_helper import RSAHelper

rsa = RSAHelper(color_perturbation=0, shape_sigma=0.1, novelty_threshold=0.8)
rsa._ensure_active_learner().verbose = False

out = open('debug_output.txt', 'w', encoding='ascii', errors='replace')

def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()

def show_concepts(label):
    log(f"\n{label}")
    for name, c in rsa.table._concepts.items():
        lab = c.mu[:3].round(3)
        log(f"  {str(name):30s}  kappa={c.kappa:.2f}  lab=[{lab[0]:.3f},{lab[1]:.3f},{lab[2]:.3f}]")

# Phase 1: Red
for epoch in range(3):
    rsa.self_train([['red box', '', '', '']])
    rsa.self_train([['red solid', '', '', '']])
    rsa.self_train([['red t', '', '', '']])
    rsa.self_train([['red l', '', '', '']])
    rsa.self_train([['red l_90', '', '', '']])
    rsa.sleep(base_rate=0.9, verbose=False)

show_concepts("=== BEFORE RED REFLECT ===")

rsa._ensure_active_learner().verbose = True
result = rsa.reflect(z_threshold=-1.2)
rsa._ensure_active_learner().verbose = False

log(f"\nRed merges: {len(result['merges'])}")
for a, b, kept in result['merges']:
    log(f"  {a} + {b} -> {kept}")

show_concepts("=== AFTER RED REFLECT ===")

# Phase 2: Blue
for epoch in range(3):
    rsa.self_train([['blue l_180', '', '', '']])
    rsa.self_train([['blue hbar', '', '', '']])
    rsa.self_train([['blue s', '', '', '']])
    rsa.self_train([['blue t', '', '', '']])
    rsa.self_train([['blue l_270', '', '', '']])
    rsa.sleep(base_rate=0.9, verbose=False)

show_concepts("=== BEFORE BLUE REFLECT ===")

rsa._ensure_active_learner().verbose = True
result = rsa.reflect(z_threshold=0)
rsa._ensure_active_learner().verbose = False
rsa.sleep(base_rate=0.3, verbose=False)

log(f"\nBlue merges: {len(result['merges'])}")
for a, b, kept in result['merges']:
    log(f"  {a} + {b} -> {kept}")

show_concepts("=== AFTER BLUE REFLECT + SLEEP ===")

# Ask
log("\n=== ASK ===")
rsa._ensure_active_learner().verbose = False
r1 = rsa.ask(['red t', 'blue hbar', '', ''], position=0)
r2 = rsa.ask(['red t', 'blue hbar', '', ''], position=1)
log(f"Ask pos=0 (red t):    is_known={r1['is_known']}, best={r1['best_token']}, z={r1['familiarity']:.2f}")
log(f"Ask pos=1 (blue hbar): is_known={r2['is_known']}, best={r2['best_token']}, z={r2['familiarity']:.2f}")

out.close()
