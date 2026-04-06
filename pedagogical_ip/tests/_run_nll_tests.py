"""Quick test for ΔNLL_local integration."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

PASS = 0
FAIL = 0

def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS: {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL: {name} -- {e}")
        FAIL += 1

def test_legacy_nll():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    r = LatticeV2Runner()
    s = r.reset(seed=42, latent_mode=True, warning_mode='fixed',
                scenario_family='fork_trap', warning_variant='legacy_bias')
    while not s.done:
        s = r.step(s)
    assert len(s.rsa_warn_diagnostics) >= 1, "no diagnostics"
    d = s.rsa_warn_diagnostics[0]
    assert 'delta_nll_local' in d, f"missing delta_nll_local: {list(d.keys())}"
    assert 'nll_before' in d
    dnll = d['delta_nll_local']
    print(f"    legacy dNLL_local={dnll:.4f}")

def test_rsa_nll():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    r = LatticeV2Runner()
    s = r.reset(seed=42, latent_mode=True, warning_mode='fixed',
                scenario_family='fork_trap', warning_variant='rsa_obs_s1')
    while not s.done:
        s = r.step(s)
    assert len(s.rsa_warn_diagnostics) >= 1
    d = s.rsa_warn_diagnostics[0]
    assert 'delta_nll_local' in d
    dnll = d['delta_nll_local']
    nll_b = d['nll_before']
    nll_a = d['nll_after']
    print(f"    S1 dNLL_local={dnll:.4f} (before={nll_b:.4f}, after={nll_a:.4f})")

def test_nll_direction():
    """Warning should decrease NLL (negative delta) for most variants."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    deltas = {}
    for v in ['legacy_bias', 'rsa_obs_s1', 'rsa_obs_l0']:
        r = LatticeV2Runner()
        s = r.reset(seed=42, latent_mode=True, warning_mode='fixed',
                    scenario_family='fork_trap', warning_variant=v)
        while not s.done:
            s = r.step(s)
        if s.rsa_warn_diagnostics:
            deltas[v] = s.rsa_warn_diagnostics[0].get('delta_nll_local', 0.0)
    print(f"    NLL deltas: {deltas}")
    # At least one variant should have negative delta (warning helps)
    any_neg = any(d < 0 for d in deltas.values())
    assert any_neg or len(deltas) == 0, "No variant improved NLL"

def test_specific_vs_generic_entropy():
    """Specific WARN (e.g. WARN_LEFT) should produce larger dH than generic."""
    from src.agents.rsa_warning_channel import (
        RSAWarningChannel, RSABeliefState, RSAUtterance,
    )
    ch = RSAWarningChannel()
    ctx = {"has_left_branch": True, "has_right_branch": True}

    # Specific: WARN_LEFT
    b_spec = RSABeliefState()
    info_spec = ch.update_belief(b_spec, RSAUtterance.WARN_LEFT, ctx, variant="s1")

    # Generic: GENERIC_WARN
    b_gen = RSABeliefState()
    info_gen = ch.update_belief(b_gen, RSAUtterance.GENERIC_WARN, ctx, variant="s1")

    dh_spec = info_spec['delta_H']
    dh_gen = info_gen['delta_H']
    print(f"    dH specific={dh_spec:.4f}, generic={dh_gen:.4f}")
    assert dh_spec > dh_gen, (
        f"Specific should produce larger dH: {dh_spec:.4f} vs {dh_gen:.4f}")

def test_held_out_family():
    """RSA warning on hazard_belt (not WARN-focused) should not crash."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    r = LatticeV2Runner()
    s = r.reset(seed=42, latent_mode=True, warning_mode='fixed',
                scenario_family='hazard_belt', warning_variant='rsa_obs_s1')
    while not s.done:
        s = r.step(s)
    # Should complete without error
    m = r.get_extended_metrics(s)
    print(f"    hazard_belt: survived={m['survived']}, warns={m['warnings']}")

def test_smoke_harness():
    """Experiment harness smoke test with dNLL_local."""
    import subprocess
    r = subprocess.run(
        ['python', 'scripts/run_step2_warning_experiment.py', '--smoke'],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"Experiment failed: {r.stderr[-300:]}"

print("=" * 60)
print("ΔNLL_local + §6.2 Tests")
print("=" * 60)

run("legacy_nll", test_legacy_nll)
run("rsa_nll", test_rsa_nll)
run("nll_direction", test_nll_direction)
run("specific_vs_generic", test_specific_vs_generic_entropy)
run("held_out_family", test_held_out_family)
run("smoke_harness", test_smoke_harness)

print("=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
