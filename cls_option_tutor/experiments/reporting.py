"""Reporting helpers for learning-increment micro benchmarks."""

from __future__ import annotations

import os

import numpy as np


def print_row(row, total: int) -> None:
    """Pretty-print one completed benchmark row."""
    idx = row.get("_job_idx", 0) + 1
    if "ERROR" in row:
        print(
            f"[{idx}/{total}] {row['task_id']} s={row['seed']} "
            f"{row['condition']} rho={row['rho_assist']} ERROR: {row['ERROR']}",
            flush=True,
        )
        return

    dsr = row["DeltaProbeSR"]
    dlsr = row.get("DeltaLocalSR", "")
    dlsr_s = f"  dLoc={dlsr:+.3f}" if isinstance(dlsr, float) else ""
    dsm = row.get("DeltaSemanticMargin", "")
    dsm_s = f"  dSM={dsm:+.3f}" if isinstance(dsm, float) else ""
    dpm = row.get("DeltaPolicyMargin", "")
    dpm_s = f"  dPM={dpm:+.3f}" if isinstance(dpm, float) else ""
    dcp = row.get("DeltaCorrectProb", "")
    dcp_s = f"  dCP={dcp:+.3f}" if isinstance(dcp, float) else ""
    print(
        f"[{idx}/{total}] {row['task_id']} s={row['seed']} "
        f"{row['condition']} rho={row['rho_assist']} => "
        f"dSR={dsr:+.3f}{dlsr_s}{dsm_s}{dpm_s}{dcp_s}  "
        f"Dmg={row['TeachDamage']}({row.get('ScriptedDamage', 0)}s)  "
        f"pSC={row.get('Protocol_SelfCorrectCount', row.get('SelfCorrectCount', 0))} "
        f"pTA={row.get('Protocol_ThenAnswerCount', row.get('ThenAnswerCount', 0))} "
        f"cueSC={row.get('PostCueGuidedSCCount', 0)} "
        f"mixOK={row.get('PostCueStructProtectCount', 0)} "
        f"cue2R={row.get('CueTrajectorySuccessWithin2RoundsRate', 0.0):.2f} "
        f"mixBD={row.get('MIXBadMassDropMean', 0.0):.3f} "
        f"mixNet={row.get('MIXDirectSelectedNetHarmDropMean', row.get('MIXOracleNetBadMassDropMean', 0.0)):.3f} "
        f"({row['ElapsedSec']:.1f}s)",
        flush=True,
    )


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05):
    """Return mean and bootstrap confidence interval for a value list."""
    if len(values) < 2:
        m = np.mean(values) if values else 0.0
        return m, m, m

    vals = np.array(values)
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(vals)), float(lo), float(hi)


def write_summary(rows, out_dir: str, ts_str: str, conditions, n_probe_global: int, n_probe_local: int) -> str | None:
    """Write markdown summary for a benchmark run and return the path."""
    md_path = os.path.join(out_dir, f"e64_micro_{ts_str}_summary.md")
    valid = [r for r in rows if "ERROR" not in r]
    if not valid:
        return None

    lines = ["# E6.4 Experience-Semantic Micro Benchmark Summary\n"]
    lines.append(f"Total runs: {len(valid)} (errors: {len(rows) - len(valid)})")
    lines.append(f"N_PROBE_GLOBAL: {n_probe_global}, N_PROBE_LOCAL: {n_probe_local}\n")

    for rho in sorted(set(r.get("rho_assist", 1.0) for r in valid)):
        lines.append(f"\n## rho_assist = {rho}\n")
        lines.append(
            "| Condition | N | dProbeSR | CI_lo | CI_hi | dLocalSR | dSemMargin | dSemRank | dPolRank | "
            "TeachDmg | ScriptDmg | DeathR | ImproveR | "
            "ProtoSC | ProtoTA | CueSC | MixOK | Cue2R |"
        )
        lines.append(
            "|-----------|---|----------|-------|-------|----------|------------|----------|----------|"
            "---------|-----------|--------|----------|"
            "--------|---------|-------|-------|-------|"
        )

        for cond in conditions:
            subset = [
                r
                for r in valid
                if r.get("condition") == cond and abs(r.get("rho_assist", 1.0) - rho) < 0.01
            ]
            if not subset:
                continue
            n = len(subset)
            dsr_vals = [r["DeltaProbeSR"] for r in subset]
            dsr_mean, dsr_lo, dsr_hi = bootstrap_ci(dsr_vals)
            dlsr_vals = [r["DeltaLocalSR"] for r in subset if isinstance(r.get("DeltaLocalSR"), (int, float))]
            dlsr_mean = np.mean(dlsr_vals) if dlsr_vals else float("nan")
            dsm_vals = [
                r["DeltaSemanticMargin"]
                for r in subset
                if isinstance(r.get("DeltaSemanticMargin"), (int, float))
            ]
            dsm_mean = np.mean(dsm_vals) if dsm_vals else float("nan")
            dsemrank_vals = [
                r["DeltaSemanticRank"]
                for r in subset
                if isinstance(r.get("DeltaSemanticRank"), (int, float))
            ]
            dsemrank_mean = np.mean(dsemrank_vals) if dsemrank_vals else float("nan")
            dpolrank_vals = [
                r["DeltaPolicyRank"]
                for r in subset
                if isinstance(r.get("DeltaPolicyRank"), (int, float))
            ]
            dpolrank_mean = np.mean(dpolrank_vals) if dpolrank_vals else float("nan")
            tdmg = np.mean([r["TeachDamage"] for r in subset])
            sdmg = np.mean([r["ScriptedDamage"] for r in subset])
            dr = np.mean([r["DeathRate"] for r in subset])
            improve_rate = np.mean([1 if r["DeltaProbeSR"] > 0 else 0 for r in subset])
            sc = sum(r.get("Protocol_SelfCorrectCount", r.get("SelfCorrectCount", 0)) for r in subset)
            ta = sum(r.get("Protocol_ThenAnswerCount", r.get("ThenAnswerCount", 0)) for r in subset)
            cue_sc = sum(r.get("PostCueGuidedSCCount", 0) for r in subset)
            mix_ok = sum(r.get("PostCueStructProtectCount", 0) for r in subset)
            cue2r = np.mean([r.get("CueTrajectorySuccessWithin2RoundsRate", 0.0) for r in subset])
            lines.append(
                f"| {cond} | {n} | {dsr_mean:+.4f} | {dsr_lo:+.4f} | {dsr_hi:+.4f} | "
                f"{'nan' if np.isnan(dlsr_mean) else f'{dlsr_mean:+.4f}'} | "
                f"{'nan' if np.isnan(dsm_mean) else f'{dsm_mean:+.4f}'} | "
                f"{'nan' if np.isnan(dsemrank_mean) else f'{dsemrank_mean:+.4f}'} | "
                f"{'nan' if np.isnan(dpolrank_mean) else f'{dpolrank_mean:+.4f}'} | "
                f"{tdmg:.1f} | {sdmg:.1f} | {dr:.3f} | {improve_rate:.2f} | "
                f"{sc} | {ta} | {cue_sc} | {mix_ok} | {cue2r:.2f} |"
            )

    lines.append("\n## Observed Wrong Reveal Risk Histogram\n")
    lines.append("| Condition | WR_Risk0 | WR_Risk1 | WR_Risk2 | WR_Risk3 | WR_Risk4 | DeathBeforeCorrect |")
    lines.append("|-----------|----------|----------|----------|----------|----------|--------------------|")
    for cond in conditions:
        subset = [r for r in valid if r.get("condition") == cond]
        if not subset:
            continue
        r0 = sum(r.get("WR_Risk0", 0) for r in subset)
        r1 = sum(r.get("WR_Risk1", 0) for r in subset)
        r2 = sum(r.get("WR_Risk2", 0) for r in subset)
        r3 = sum(r.get("WR_Risk3", 0) for r in subset)
        r4 = sum(r.get("WR_Risk4", 0) for r in subset)
        dbc = sum(r.get("DeathBeforeCorrect", 0) for r in subset)
        lines.append(f"| {cond} | {r0} | {r1} | {r2} | {r3} | {r4} | {dbc} |")

    if any(any(r.get(f"Protocol_WR_Risk{k}", 0) for k in range(5)) for r in valid):
        lines.append("\n## Protocol Wrong Reveal Risk Histogram\n")
        lines.append("| Condition | P_WR_Risk0 | P_WR_Risk1 | P_WR_Risk2 | P_WR_Risk3 | P_WR_Risk4 | P_DeathBeforeCorrect |")
        lines.append("|-----------|------------|------------|------------|------------|------------|----------------------|")
        for cond in conditions:
            subset = [r for r in valid if r.get("condition") == cond]
            if not subset:
                continue
            r0 = sum(r.get("Protocol_WR_Risk0", 0) for r in subset)
            r1 = sum(r.get("Protocol_WR_Risk1", 0) for r in subset)
            r2 = sum(r.get("Protocol_WR_Risk2", 0) for r in subset)
            r3 = sum(r.get("Protocol_WR_Risk3", 0) for r in subset)
            r4 = sum(r.get("Protocol_WR_Risk4", 0) for r in subset)
            dbc = sum(r.get("Protocol_DeathBeforeCorrect", 0) for r in subset)
            lines.append(f"| {cond} | {r0} | {r1} | {r2} | {r3} | {r4} | {dbc} |")

    lines.append("\n## MIX Target Audit\n")
    lines.append("| Condition | MIXRate | MixBDrop | MixRemovedBad | MixRemovedProb | DirectNetH | DirectReg | BanHighRisk | BanLastWrong | BanTopWrong | BanPolicyMass | TargetRegretNet | JointGate | JointRepl | JointRegret | JointIntReg | BadWAIT+Cue |")
    lines.append("|-----------|---------|----------|----------------|----------------|------------|-----------|-------------|--------------|-------------|---------------|-----------------|-----------|-----------|-------------|-------------|------------|")
    for cond in conditions:
        subset = [r for r in valid if r.get("condition") == cond]
        if not subset:
            continue
        mix_rate = np.mean([r.get("PostReveal_MIXRate", 0.0) for r in subset])
        mix_bdrop = np.mean([r.get("MIXBadMassDropMean", 0.0) for r in subset])
        mix_removed_bad = np.mean([r.get("MIXRemovedBadMassMean", 0.0) for r in subset])
        mix_removed_prob = np.mean([r.get("MIXRemovedProbMassMean", 0.0) for r in subset])
        direct_net = np.mean([r.get("MIXDirectSelectedNetHarmDropMean", 0.0) for r in subset])
        direct_reg = np.mean([r.get("MIXDirectNetTargetRegretMean", 0.0) for r in subset])
        ban_hr = np.mean([r.get("MIXBanTargetWasHighRiskRate", 0.0) for r in subset])
        ban_lw = np.mean([r.get("MIXBanTargetWasLastWrongRate", 0.0) for r in subset])
        ban_top = np.mean([r.get("MIXBanTargetWasTopProbWrongRate", 0.0) for r in subset])
        ban_mass = np.mean([r.get("MIXBanTargetMeanPolicyMass", 0.0) for r in subset])
        regret_net = np.mean([r.get("MIXNetTargetRegretMean", 0.0) for r in subset])
        joint_gate = np.mean([r.get("MIXJointGateAppliedRate", 0.0) for r in subset])
        joint_repl = np.mean([r.get("MIXJointGateReplacedRate", 0.0) for r in subset])
        joint_regret = np.mean([r.get("MIXJointRegretMean", 0.0) for r in subset])
        joint_int_regret = np.mean([r.get("MIXJointInteractionRegretMean", 0.0) for r in subset])
        bad_wait = np.mean([r.get("BadWAIT_PostReveal_PositiveCueRate", 0.0) for r in subset])
        lines.append(
            f"| {cond} | {mix_rate:.4f} | {mix_bdrop:.4f} | {mix_removed_bad:.4f} | {mix_removed_prob:.4f} | "
            f"{direct_net:.4f} | {direct_reg:.4f} | {ban_hr:.4f} | {ban_lw:.4f} | {ban_top:.4f} | {ban_mass:.4f} | {regret_net:.4f} | "
            f"{joint_gate:.4f} | {joint_repl:.4f} | {joint_regret:.4f} | {joint_int_regret:.4f} | {bad_wait:.4f} |"
        )

    lines.append("\n## Post-Cue Wrong Taxonomy\n")
    lines.append("| Condition | Wrong | SameWrong | DiffSafeDiag | Bounded | HighRisk | FarWrong | Refresh | Grace2R |")
    lines.append("|-----------|-------|-----------|--------------|---------|----------|----------|---------|---------|")
    for cond in conditions:
        subset = [r for r in valid if r.get("condition") == cond]
        if not subset:
            continue
        wrong = np.mean([r.get("PostCueWrongPickRate", 0.0) for r in subset])
        same = np.mean([r.get("PostCueWrongPick_SameWrongRate", 0.0) for r in subset])
        diff_safe = np.mean([r.get("PostCueWrongPick_DifferentSafeDiagRate", 0.0) for r in subset])
        bounded = np.mean([r.get("PostCueWrongPick_BoundedDiagRate", 0.0) for r in subset])
        high = np.mean([r.get("PostCueWrongPick_HighRiskRate", 0.0) for r in subset])
        far = np.mean([r.get("PostCueWrongPick_FarWrongRate", 0.0) for r in subset])
        refresh = np.mean([r.get("PostCueRefreshRate", 0.0) for r in subset])
        grace2 = np.mean([r.get("CueTrajectorySuccessWithin2RoundsRate", 0.0) for r in subset])
        lines.append(
            f"| {cond} | {wrong:.4f} | {same:.4f} | {diff_safe:.4f} | {bounded:.4f} | {high:.4f} | {far:.4f} | {refresh:.4f} | {grace2:.4f} |"
        )

    lines.append("\n## Allow / Productive Reveal\n")
    lines.append("| Condition | AllowWAIT | SafeDiagAfterAllow | SafeDiag->Cue | SafeDiag->Grace | SafeDiag->2RSuccess | Bounded->2RSuccess |")
    lines.append("|-----------|-----------|--------------------|---------------|-----------------|---------------------|-------------------|")
    for cond in conditions:
        subset = [r for r in valid if r.get("condition") == cond]
        if not subset:
            continue
        allow_n = np.mean([r.get("AllowSafeDiagDecisionCount", 0.0) for r in subset])
        allow_safe = np.mean([r.get("SafeDiagRevealAfterAllowRate", 0.0) for r in subset])
        allow_cue = np.mean([r.get("SafeDiagRevealThenCueRate", 0.0) for r in subset])
        allow_grace = np.mean([r.get("SafeDiagRevealThenGraceRate", 0.0) for r in subset])
        allow_succ = np.mean([r.get("SafeDiagRevealThenTrajectorySuccessRate", 0.0) for r in subset])
        bounded_succ = np.mean([r.get("BoundedRevealThenTrajectorySuccessRate", 0.0) for r in subset])
        lines.append(
            f"| {cond} | {allow_n:.2f} | {allow_safe:.4f} | {allow_cue:.4f} | {allow_grace:.4f} | {allow_succ:.4f} | {bounded_succ:.4f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary: {md_path}")
    return md_path
