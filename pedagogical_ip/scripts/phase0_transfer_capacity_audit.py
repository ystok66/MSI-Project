"""
Phase 0 — Q2: Transfer Capacity Audit.

Tests whether switching from linear (4D→1D) to structured basis (5D→1D risk,
4D→1D cost) unlocks non-zero StateGain.

Shadow predictor is defined ENTIRELY within this script. No promotion.

Usage:
    python scripts/phase0_transfer_capacity_audit.py [--sessions 6] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from collections import defaultdict
from copy import deepcopy

from src.agents.risk_model import BayesianRiskHead, _sigmoid
from src.agents.cost_risk_model import BayesianCostHead


# ═══════════════════════════════════════════════════════════
# Shadow Structured Basis Head (NOT promoted to src/)
# ═══════════════════════════════════════════════════════════

def basis_risk(z):
    """φ_r(z) = [1, z₂, z₃, z₂z₃, |z₂-z₃|]"""
    return np.array([1.0, z[2], z[3], z[2] * z[3], abs(z[2] - z[3])])


def basis_cost(z):
    """φ_c(z) = [1, z₀, z₁, z₂+z₃]"""
    return np.array([1.0, z[0], z[1], z[2] + z[3]])


class StructuredBasisRiskHead:
    """Shadow-only 5D basis risk head."""

    def __init__(self, learning_rate=0.3, prior_var=1.0):
        self.d = 5
        self.w = np.zeros(self.d, dtype=np.float64)
        self.b = 0.0
        self.prior_var = prior_var
        self.lr = learning_rate
        self.n_updates = 0
        self.xx_sum = np.zeros((self.d, self.d), dtype=np.float64)

    def predict_risk(self, x):
        phi = basis_risk(x)
        logit = self.w @ phi + self.b
        return float(_sigmoid(np.array([logit]))[0])

    def predict_uncertainty(self, x):
        if self.n_updates < 2:
            return 0.25
        phi = basis_risk(x)
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            p = self.predict_risk(x)
            return float(p * (1 - p) * (1 + phi @ H_inv @ phi))
        except np.linalg.LinAlgError:
            return 0.25

    def update_from_label(self, x, y, weight=1.0):
        phi = basis_risk(x)
        p = self.predict_risk(x)
        error = y - p
        grad_w = -error * phi * weight + self.w / self.prior_var
        grad_b = -error * weight
        grad_norm = float(np.linalg.norm(grad_w))
        if not np.isfinite(grad_norm):
            return
        if grad_norm > 5.0:
            grad_w *= 5.0 / grad_norm
        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -5.0, 5.0))
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > 10.0:
            self.w *= 10.0 / w_norm
        self.xx_sum += weight * np.outer(phi, phi)
        self.n_updates += 1

    def reset(self):
        self.w[:] = 0.0
        self.b = 0.0
        self.xx_sum[:] = 0.0
        self.n_updates = 0


class StructuredBasisCostHead:
    """Shadow-only 4D basis cost head."""

    def __init__(self, learning_rate=0.1, prior_var=1.0):
        self.d = 4
        self.w = np.zeros(self.d, dtype=np.float64)
        self.b = 1.0
        self.prior_var = prior_var
        self.lr = learning_rate
        self.n_updates = 0
        self.xx_sum = np.zeros((self.d, self.d), dtype=np.float64)

    def predict_cost(self, x):
        phi = basis_cost(x)
        return float(max(self.w @ phi + self.b, 0.1))

    def predict_uncertainty(self, x):
        if self.n_updates < 2:
            return 1.0
        phi = basis_cost(x)
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            return float(max(phi @ H_inv @ phi, 0.01))
        except np.linalg.LinAlgError:
            return 1.0

    def update_from_label(self, x, y, weight=1.0):
        phi = basis_cost(x)
        pred = self.w @ phi + self.b
        error = y - pred
        grad_w = -error * phi * weight + self.w / self.prior_var
        grad_b = -error * weight
        grad_norm = float(np.linalg.norm(grad_w))
        if not np.isfinite(grad_norm):
            return
        if grad_norm > 5.0:
            grad_w *= 5.0 / grad_norm
        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -5.0, 5.0))
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > 10.0:
            self.w *= 10.0 / w_norm
        self.xx_sum += weight * np.outer(phi, phi)
        self.n_updates += 1

    def reset(self):
        self.w[:] = 0.0
        self.b = 1.0
        self.xx_sum[:] = 0.0
        self.n_updates = 0


class StructuredBasisHead:
    """Shadow CostRisk head using structured basis. Implements latent_predictor protocol."""

    def __init__(self):
        self.cost_head = StructuredBasisCostHead()
        self.risk_head = StructuredBasisRiskHead()
        self.risk_supervision = "oracle_visited"

    def predict_cost(self, x):
        return self.cost_head.predict_cost(x)

    def predict_risk(self, x):
        return self.risk_head.predict_risk(x)

    def predict_cost_uncertainty(self, x):
        return self.cost_head.predict_uncertainty(x)

    def predict_risk_uncertainty(self, x):
        return self.risk_head.predict_uncertainty(x)

    def predict_cost_uncertainty_from_var(self, x_var):
        # fallback: use raw Hessian
        return self.predict_cost_uncertainty(np.sqrt(x_var + 1e-8))

    def predict_risk_uncertainty_from_var(self, x_var):
        return self.predict_risk_uncertainty(np.sqrt(x_var + 1e-8))

    def update_from_outcome(self, x, cost_label, risk_label, weight=1.0):
        self.cost_head.update_from_label(x, cost_label, weight=weight)
        self.risk_head.update_from_label(x, risk_label, weight=weight)

    def reset(self):
        self.cost_head.reset()
        self.risk_head.reset()

    @property
    def n_updates(self):
        return self.risk_head.n_updates


# ═══════════════════════════════════════════════════════════
# PRS Session Runner (minimal, using existing PRSSession)
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 Q2: Transfer capacity audit")
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="Smoke: 1 session per condition")
    p.add_argument("--output-dir", default="results/phase0")
    return p.parse_args()


def run_prs_session(session_seed, model_type, stateful, weight_mode="session_shared"):
    """Run one PRS session with given model type and statefulness."""
    from src.envs.prs_session import PRSSession, SessionConfig
    from src.agents.cost_risk_model import LatentCostRiskHead

    cfg = SessionConfig(
        session_seed=session_seed,
        weight_mode=weight_mode,
        tutor_strategy="selective",
        curriculum="gtet_only",  # GTET is more stable for capacity test
        difficulty="medium",
        persist_agent_memory=stateful,
        block_a_size=20,  # shorter for audit
        block_b_size=10,
        block_c_size=10,
        block_d_size=10,
    )

    session = PRSSession(config=cfg)

    # Monkey-patch the latent predictor if using basis shadow
    if model_type == "basis_shadow":
        _orig_run = session._run_episode

        def _patched_run(runner, state, ep_spec, tutor_enabled):
            # Replace the latent predictor with our shadow head
            if state.latent_predictor is not None:
                # Create a basis head with same weights if stateful
                if not hasattr(session, '_shadow_head'):
                    session._shadow_head = StructuredBasisHead()
                state.latent_predictor = session._shadow_head
                if not stateful:
                    state.latent_predictor.reset()
            return _orig_run(runner, state, ep_spec, tutor_enabled)

        session._run_episode = _patched_run

    try:
        results = session.run_session()
        return results
    except Exception as e:
        return {"error": str(e)}


def compute_block_tbsr(block_results):
    """Compute TBSR from block episode results."""
    if not block_results:
        return 0.0
    return float(np.mean([r.get("success", False) for r in block_results]))


def main():
    args = parse_args()
    n_sessions = 1 if args.smoke else args.sessions
    
    lines = []
    lines.append("Phase 0 — Q2: Transfer Capacity Audit")
    lines.append(f"  sessions_per_condition={n_sessions}")
    lines.append("=" * 80)

    # 4 conditions: linear×{stateless,stateful}, basis×{stateless,stateful}
    conditions = [
        ("linear_current", False),
        ("linear_current", True),
        ("basis_shadow", False),
        ("basis_shadow", True),
    ]

    results_by_cond = {}

    for model_type, stateful in conditions:
        cond_name = f"{model_type}_{'stateful' if stateful else 'stateless'}"
        lines.append(f"\n--- Condition: {cond_name} ---")

        block_tbsrs = defaultdict(list)

        for i in range(n_sessions):
            session_seed = 42 + i * 7
            try:
                res = run_prs_session(session_seed, model_type, stateful)
                if "error" in res:
                    lines.append(f"  session {i}: ERROR: {res['error']}")
                    continue

                for block_id in ["A", "B", "C", "D"]:
                    br = res.get("block_results", {}).get(block_id, [])
                    tbsr = compute_block_tbsr(br)
                    block_tbsrs[block_id].append(tbsr)

                lines.append(f"  session {i}: A={block_tbsrs['A'][-1]:.3f} "
                             f"B={block_tbsrs['B'][-1]:.3f} "
                             f"C={block_tbsrs['C'][-1]:.3f} "
                             f"D={block_tbsrs['D'][-1]:.3f}")
            except Exception as e:
                lines.append(f"  session {i}: EXCEPTION: {e}")

        results_by_cond[cond_name] = dict(block_tbsrs)

    # === StateGain computation ===
    lines.append(f"\n{'='*80}")
    lines.append("STATE GAIN ANALYSIS")
    lines.append("=" * 80)

    for model_type in ["linear_current", "basis_shadow"]:
        stateful_name = f"{model_type}_stateful"
        stateless_name = f"{model_type}_stateless"

        lines.append(f"\n  Model: {model_type}")
        for block_id in ["B", "C", "D"]:
            sf = results_by_cond.get(stateful_name, {}).get(block_id, [])
            sl = results_by_cond.get(stateless_name, {}).get(block_id, [])
            if sf and sl:
                sg = np.mean(sf) - np.mean(sl)
                lines.append(f"    Block {block_id}: stateful={np.mean(sf):.3f}, "
                             f"stateless={np.mean(sl):.3f}, StateGain={sg:+.3f}")
            else:
                lines.append(f"    Block {block_id}: insufficient data")

    # === Verdict ===
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Check if basis has positive StateGain
    basis_gains = []
    for block_id in ["B", "C", "D"]:
        sf = results_by_cond.get("basis_shadow_stateful", {}).get(block_id, [])
        sl = results_by_cond.get("basis_shadow_stateless", {}).get(block_id, [])
        if sf and sl:
            basis_gains.append(np.mean(sf) - np.mean(sl))

    linear_gains = []
    for block_id in ["B", "C", "D"]:
        sf = results_by_cond.get("linear_current_stateful", {}).get(block_id, [])
        sl = results_by_cond.get("linear_current_stateless", {}).get(block_id, [])
        if sf and sl:
            linear_gains.append(np.mean(sf) - np.mean(sl))

    if basis_gains:
        avg_basis_sg = np.mean(basis_gains)
        avg_linear_sg = np.mean(linear_gains) if linear_gains else 0.0

        if avg_basis_sg <= 0.02:
            lines.append("VERDICT A: Basis 仍然没有 transfer。")
            lines.append(f"  avg basis StateGain = {avg_basis_sg:+.3f}")
            lines.append(f"  avg linear StateGain = {avg_linear_sg:+.3f}")
            lines.append("  → 当前失败不是'线性太简单'，而是单 episode 数据已足够。")
            lines.append("  → '继续调 lr/线性头' 正式结束。")
        elif avg_basis_sg > 0.05:
            lines.append("VERDICT B: Basis 开始出现稳定的正 StateGain。")
            lines.append(f"  avg basis StateGain = {avg_basis_sg:+.3f}")
            lines.append(f"  avg linear StateGain = {avg_linear_sg:+.3f}")
            lines.append("  → 表达能力确实是瓶颈的一部分。")
            lines.append("  → Phase 1 值得进入 slow-fast / dual-timescale。")
        else:
            lines.append("VERDICT: 边界区域，需要更大样本量确认。")
            lines.append(f"  avg basis StateGain = {avg_basis_sg:+.3f}")
    else:
        lines.append("VERDICT: 数据不足，无法判定。")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "q2_transfer_capacity_audit.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
