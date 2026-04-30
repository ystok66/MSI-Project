"""Utilities for validating family-benchmark candidate pools.

This module enforces a stricter acceptance pipeline than the older
structure-only validators:

1. structure sanity
2. exact render consistency under the real repository semantics
3. runtime family validation under a chosen tutor condition
4. formal slice assembly from accepted tasks only
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.experiments.condition_overrides import (
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.metrics_extractors import build_allow_family_audit
from cls_option_tutor.experiments.run_learning_increment_micro import (
    _apply_condition_overrides,
    make_cfg,
)
from cls_option_tutor.grammar.task_adapter import parse_task_file
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent


INTENDED_TO_RATE_KEY: Dict[str, str] = {
    "ALLOW_CRITICAL_HEAVY": "NativeLikeAllowRate",
    "MIXED_PROD_HARM_HEAVY": "MixedProdHarmRate",
    "PROTECT_CRITICAL_HEAVY": "ProtectCriticalRate",
    "BORING_MASTERY_HEAVY": "BoringMasteryRate",
}

INTENDED_TO_RUNTIME_FAMILY: Dict[str, str] = {
    "ALLOW_CRITICAL_HEAVY": "NATIVE_LIKE_ALLOW",
    "MIXED_PROD_HARM_HEAVY": "MIXED_PROD_HARM",
    "PROTECT_CRITICAL_HEAVY": "PROTECT_CRITICAL",
    "BORING_MASTERY_HEAVY": "BORING_MASTERY",
}

CORE_RATE_KEYS: Sequence[str] = (
    "NativeLikeAllowRate",
    "MixedProdHarmRate",
    "ProtectCriticalRate",
    "BoringMasteryRate",
)


@dataclass(frozen=True)
class CandidatePool:
    root: Path
    tasks_dir: Path
    manifest_csv: Path
    manifest_jsonl: Optional[Path]


def _mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return mean(vals) if vals else 0.0


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def discover_candidate_pool(pool_dir: str | Path) -> CandidatePool:
    root = Path(pool_dir).resolve()
    tasks_dir = root / "tasks"
    manifest_csv = root / "family_manifest.csv"
    manifest_jsonl = root / "family_manifest.jsonl"

    if not root.exists():
        raise FileNotFoundError(f"candidate pool not found: {root}")
    if not tasks_dir.exists():
        raise FileNotFoundError(f"tasks directory not found: {tasks_dir}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_csv}")

    return CandidatePool(
        root=root,
        tasks_dir=tasks_dir,
        manifest_csv=manifest_csv,
        manifest_jsonl=manifest_jsonl if manifest_jsonl.exists() else None,
    )


def load_manifest_rows(pool: CandidatePool) -> List[dict]:
    with pool.manifest_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {pool.manifest_csv}")
    return rows


def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in ("*SUPPORT*", "*QUERY*", "*GRAMMAR*"):
            current = line.strip("*")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def validate_structure_for_row(pool: CandidatePool, row: Mapping[str, str]) -> dict:
    filename = str(row["filename"])
    task_path = pool.root / filename
    errors: List[str] = []

    if not task_path.exists():
        return {
            "task_id": row["task_id"],
            "filename": filename,
            "intended_family": row["intended_family"],
            "structure_valid": False,
            "structure_error_count": 1,
            "structure_errors": f"missing file: {filename}",
        }

    text = task_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    for name in ("SUPPORT", "QUERY", "GRAMMAR"):
        if name not in sections:
            errors.append(f"missing {name}")

    support = sections.get("SUPPORT", [])
    query = sections.get("QUERY", [])
    grammar = sections.get("GRAMMAR", [])

    try:
        n_nouns = int(row.get("n_nouns", "0") or 0)
    except ValueError:
        n_nouns = 0
    try:
        n_ops = int(row.get("n_operators", "0") or 0)
    except ValueError:
        n_ops = 0

    if support:
        if not (12 <= len(support) <= 20):
            errors.append(f"support count {len(support)}")
    if query:
        if not (8 <= len(query) <= 16):
            errors.append(f"query count {len(query)}")
    if grammar and n_nouns > 0 and n_ops > 0 and len(grammar) != (n_nouns + n_ops):
        errors.append(f"grammar count {len(grammar)} expected {n_nouns + n_ops}")

    for line in support + query:
        if not line.startswith("IN: ") or " OUT: " not in line:
            errors.append(f"bad IO line: {line}")
    for line in grammar:
        if "->" not in line:
            errors.append(f"bad grammar line: {line}")

    return {
        "task_id": row["task_id"],
        "filename": filename,
        "intended_family": row["intended_family"],
        "structure_valid": len(errors) == 0,
        "structure_error_count": len(errors),
        "structure_errors": " || ".join(errors[:10]),
    }


def validate_exact_render_for_row(
    pool: CandidatePool,
    row: Mapping[str, str],
    *,
    rho_assist: float,
) -> dict:
    task_id = str(row["task_id"])
    filename = str(row["filename"])
    generator_mode = str(row.get("generator_mode", "v2_overlap"))

    cfg = make_cfg(
        n_sup=4,
        rho_assist=rho_assist,
        generator_mode=generator_mode,
        tutor_lg_mode="off",
        highlight_mode="diagnostic",
    )
    env = OptionEnv(cfg=cfg, data_dir=str(pool.tasks_dir))
    support, query, grammar = env.adapter.load_task(task_id)

    support_bad = 0
    query_bad = 0
    any_none_render = False
    first_support_bad = None
    first_query_bad = None

    for idx, ex in enumerate(support):
        rendered = env.adapter.render(ex.words, grammar)
        if rendered is None:
            any_none_render = True
        if rendered != ex.output:
            support_bad += 1
            if first_support_bad is None:
                first_support_bad = (idx, ex.words, ex.output, rendered)

    for idx, ex in enumerate(query):
        rendered = env.adapter.render(ex.words, grammar)
        if rendered is None:
            any_none_render = True
        if rendered != ex.output:
            query_bad += 1
            if first_query_bad is None:
                first_query_bad = (idx, ex.words, ex.output, rendered)

    exact_valid = (support_bad == 0) and (query_bad == 0) and (not any_none_render)

    def _serialize_bad(example):
        if example is None:
            return ""
        idx, words, expected, rendered = example
        rendered_str = "None" if rendered is None else " ".join(rendered)
        return (
            f"idx={idx}; IN={' '.join(words)}; "
            f"expected={' '.join(expected)}; rendered={rendered_str}"
        )

    return {
        "task_id": task_id,
        "filename": filename,
        "intended_family": str(row["intended_family"]),
        "generator_mode": generator_mode,
        "support_count": len(support),
        "query_count": len(query),
        "support_mismatch_count": support_bad,
        "query_mismatch_count": query_bad,
        "any_none_render": any_none_render,
        "exact_render_valid": exact_valid,
        "first_support_mismatch": _serialize_bad(first_support_bad),
        "first_query_mismatch": _serialize_bad(first_query_bad),
    }


def _run_teach_block_for_task(
    pool: CandidatePool,
    row: Mapping[str, str],
    *,
    seed: int,
    condition: str,
    rho_assist: float,
):
    condition_eff = resolve_condition_alias(condition)
    cfg = make_cfg(
        n_sup=4,
        rho_assist=rho_assist,
        generator_mode=str(row.get("generator_mode", "v2_overlap")),
        tutor_lg_mode="off",
        highlight_mode="diagnostic",
    )
    cfg = _apply_condition_overrides(copy.deepcopy(cfg), condition_eff)

    env = OptionEnv(cfg=cfg, data_dir=str(pool.tasks_dir))
    learner = LearnerAgent(cfg=cfg, seed=seed)
    support, _, grammar = env.adapter.load_task(str(row["task_id"]))
    init_block = env.reset_block(str(row["task_id"]), seed=seed)
    learner.init_block(init_block, grammar, support)

    if condition_eff.startswith("script_") or condition_eff.startswith("no_tutor_"):
        protocol = extract_scripted_protocol_name(condition_eff)
        runner = ScriptedProtocolRunner(cfg=cfg, protocol=protocol)
        result = runner.run_block(
            OptionEnv(cfg=cfg, data_dir=str(pool.tasks_dir)),
            learner,
            str(row["task_id"]),
            seed=seed,
        )
        return result.block

    tutor = SparseTutorAgent(cfg=cfg)
    return tutor.run_block(
        OptionEnv(cfg=cfg, data_dir=str(pool.tasks_dir)),
        learner,
        str(row["task_id"]),
        seed=seed,
    )


def summarize_runtime_family_rows(rows: Sequence[dict]) -> dict:
    family_counts: Dict[str, int] = {}
    for row in rows:
        fam = str(row.get("family_split", "UNKNOWN"))
        family_counts[fam] = family_counts.get(fam, 0) + 1

    stats = {
        "StateCount": len(rows),
        "NativeLikeAllowRate": _mean_or_zero(
            1.0 if str(r.get("family_split", "")) == "NATIVE_LIKE_ALLOW" else 0.0
            for r in rows
        ),
        "MixedProdHarmRate": _mean_or_zero(
            1.0 if str(r.get("family_split", "")) == "MIXED_PROD_HARM" else 0.0
            for r in rows
        ),
        "ProtectCriticalRate": _mean_or_zero(
            1.0 if str(r.get("family_split", "")) == "PROTECT_CRITICAL" else 0.0
            for r in rows
        ),
        "BoringMasteryRate": _mean_or_zero(
            1.0 if str(r.get("family_split", "")) == "BORING_MASTERY" else 0.0
            for r in rows
        ),
        "AllowPreserveRate": _mean_or_zero(
            1.0 if bool(r.get("allow_preserved", False)) else 0.0 for r in rows
        ),
        "ProductiveRevealRate": _mean_or_zero(
            1.0 if bool(r.get("productive_reveal_after_state", False)) else 0.0
            for r in rows
        ),
        "LoopCompleteRate": _mean_or_zero(
            1.0 if bool(r.get("loop_complete_after_state", False)) else 0.0
            for r in rows
        ),
        "DeathBeforeCorrectRate": _mean_or_zero(
            1.0 if bool(r.get("death_before_correct_after_state", False)) else 0.0
            for r in rows
        ),
        "MeanDamageAfterState": _mean_or_zero(
            float(r.get("damage_after_state", 0.0)) for r in rows
        ),
        "MeanPProd": _mean_or_zero(float(r.get("p_prod_total", 0.0)) for r in rows),
        "MeanHarmMass": _mean_or_zero(float(r.get("harm_mass", 0.0)) for r in rows),
        "MeanSafeDiagQualityGap": _mean_or_zero(
            float(r.get("safe_diag_quality_gap", 0.0)) for r in rows
        ),
        "MeanPcorrectWAIT": _mean_or_zero(
            float(r.get("p_correct_wait", 0.0)) for r in rows
        ),
    }
    stats.update({f"Count_{k}": v for k, v in family_counts.items()})
    return stats


def validate_runtime_family_for_row(
    pool: CandidatePool,
    row: Mapping[str, str],
    *,
    seeds: Sequence[int],
    condition: str,
    rho_assist: float,
) -> dict:
    all_rows: List[dict] = []
    for seed in seeds:
        block = _run_teach_block_for_task(
            pool,
            row,
            seed=seed,
            condition=condition,
            rho_assist=rho_assist,
        )
        for audit_row in build_allow_family_audit(block):
            audit_row = dict(audit_row)
            audit_row.setdefault("task_id", str(row["task_id"]))
            audit_row["seed"] = seed
            all_rows.append(audit_row)

    stats = summarize_runtime_family_rows(all_rows)
    rate_values = {key: float(stats.get(key, 0.0)) for key in CORE_RATE_KEYS}
    dominant_rate_key = max(rate_values, key=rate_values.get) if rate_values else ""
    dominant_rate = rate_values.get(dominant_rate_key, 0.0)
    dominant_runtime_family = ""
    if dominant_rate_key == "NativeLikeAllowRate":
        dominant_runtime_family = "NATIVE_LIKE_ALLOW"
    elif dominant_rate_key == "MixedProdHarmRate":
        dominant_runtime_family = "MIXED_PROD_HARM"
    elif dominant_rate_key == "ProtectCriticalRate":
        dominant_runtime_family = "PROTECT_CRITICAL"
    elif dominant_rate_key == "BoringMasteryRate":
        dominant_runtime_family = "BORING_MASTERY"

    intended_family = str(row["intended_family"])
    target_rate_key = INTENDED_TO_RATE_KEY.get(intended_family, "")
    target_runtime_family = INTENDED_TO_RUNTIME_FAMILY.get(intended_family, "")

    result = {
        "task_id": str(row["task_id"]),
        "filename": str(row["filename"]),
        "intended_family": intended_family,
        "generator_mode": str(row.get("generator_mode", "v2_overlap")),
        "condition": condition,
        "seed_count": len(seeds),
        "target_rate_key": target_rate_key,
        "target_runtime_family": target_runtime_family,
        "dominant_rate_key": dominant_rate_key,
        "dominant_runtime_family": dominant_runtime_family,
        **stats,
    }
    return result


def choose_formal_slice_acceptance(
    runtime_row: Mapping[str, object],
    *,
    min_target_rate: float,
) -> tuple[bool, str]:
    target_rate_key = str(runtime_row.get("target_rate_key", ""))
    target_runtime_family = str(runtime_row.get("target_runtime_family", ""))
    if not target_rate_key or not target_runtime_family:
        return False, "unknown_intended_family"

    target_rate = float(runtime_row.get(target_rate_key, 0.0) or 0.0)
    dominant_runtime_family = str(runtime_row.get("dominant_runtime_family", ""))
    state_count = int(runtime_row.get("StateCount", 0) or 0)

    if state_count <= 0:
        return False, "no_runtime_states"
    if target_rate < min_target_rate:
        return False, "target_rate_below_threshold"
    if dominant_runtime_family != target_runtime_family:
        return False, "target_family_not_dominant"
    return True, "accepted"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def assemble_formal_slice(
    pool: CandidatePool,
    manifest_rows: Sequence[Mapping[str, str]],
    exact_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    *,
    min_target_rate: float,
    out_dir: Path,
) -> dict:
    exact_by_task = {str(r["task_id"]): r for r in exact_rows}
    runtime_by_task = {str(r["task_id"]): r for r in runtime_rows}

    accepted_rows: List[dict] = []
    rejected_rows: List[dict] = []
    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for manifest_row in manifest_rows:
        task_id = str(manifest_row["task_id"])
        exact_row = exact_by_task.get(task_id)
        runtime_row = runtime_by_task.get(task_id)
        exact_valid = bool(exact_row and exact_row.get("exact_render_valid", False))

        accepted = False
        reason = "exact_render_invalid"
        if exact_valid and runtime_row is not None:
            accepted, reason = choose_formal_slice_acceptance(
                runtime_row, min_target_rate=min_target_rate
            )

        record = {
            **dict(manifest_row),
            "exact_render_valid": exact_valid,
            "accepted_for_formal_slice": accepted,
            "formal_slice_reason": reason,
        }
        if runtime_row is not None:
            for key in (
                "StateCount",
                "NativeLikeAllowRate",
                "MixedProdHarmRate",
                "ProtectCriticalRate",
                "BoringMasteryRate",
                "AllowPreserveRate",
                "ProductiveRevealRate",
                "LoopCompleteRate",
                "MeanPProd",
                "MeanHarmMass",
                "MeanSafeDiagQualityGap",
                "MeanPcorrectWAIT",
                "dominant_runtime_family",
            ):
                record[key] = runtime_row.get(key, "")

        if accepted:
            src = pool.root / str(manifest_row["filename"])
            dst = tasks_dir / src.name
            shutil.copy2(src, dst)
            accepted_rows.append(record)
        else:
            rejected_rows.append(record)

    write_csv(out_dir / "formal_slice_manifest.csv", accepted_rows)
    write_csv(out_dir / "formal_slice_rejected.csv", rejected_rows)

    by_family: Dict[str, int] = {}
    for row in accepted_rows:
        fam = str(row.get("intended_family", "UNKNOWN"))
        by_family[fam] = by_family.get(fam, 0) + 1

    summary_lines = [
        "# Formal Family Benchmark Slice",
        "",
        f"- Source pool: `{pool.root}`",
        f"- Accepted tasks: `{len(accepted_rows)}`",
        f"- Rejected tasks: `{len(rejected_rows)}`",
        f"- `min_target_rate`: `{min_target_rate}`",
        "",
        "## Accepted counts by intended family",
        "",
    ]
    if by_family:
        for fam in sorted(by_family):
            summary_lines.append(f"- `{fam}`: `{by_family[fam]}`")
    else:
        summary_lines.append("- none")
    summary_lines.append("")
    summary_lines.append("## Acceptance rule")
    summary_lines.append("")
    summary_lines.append(
        "A task enters the formal slice only if it is exact-render valid, has "
        "runtime family states, clears the target-rate threshold, and its "
        "target runtime family is the dominant core family."
    )
    (out_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
    }


def build_validation_report(
    *,
    pool: CandidatePool,
    structure_rows: Sequence[Mapping[str, object]],
    exact_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    formal_summary: Mapping[str, object],
    condition: str,
    seeds: Sequence[int],
    min_target_rate: float,
) -> str:
    structure_valid = sum(
        1 for row in structure_rows if bool(row.get("structure_valid", False))
    )
    exact_valid = sum(
        1 for row in exact_rows if bool(row.get("exact_render_valid", False))
    )

    lines = [
        "# Family Candidate Pool Validation Report",
        "",
        f"- Pool: `{pool.root}`",
        f"- Condition: `{condition}`",
        f"- Seeds: `{', '.join(str(s) for s in seeds)}`",
        f"- `min_target_rate`: `{min_target_rate}`",
        "",
        "## Stage 1: Structure",
        "",
        f"- Structure-valid tasks: `{structure_valid} / {len(structure_rows)}`",
        "",
        "## Stage 2: Exact render",
        "",
        f"- Exact-render-valid tasks: `{exact_valid} / {len(exact_rows)}`",
        "",
    ]

    if runtime_rows:
        lines.extend([
            "## Stage 3: Runtime family validation",
            "",
            "| Task | Intended | TargetRate | DominantRuntimeFamily | NativeLikeAllowRate | MixedProdHarmRate | ProtectCriticalRate | BoringMasteryRate | LoopCompleteRate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in sorted(runtime_rows, key=lambda r: str(r.get("task_id", ""))):
            target_key = str(row.get("target_rate_key", ""))
            target_rate = float(row.get(target_key, 0.0) or 0.0)
            lines.append(
                "| "
                + " | ".join([
                    str(row.get("task_id", "")),
                    str(row.get("intended_family", "")),
                    _fmt(target_rate),
                    str(row.get("dominant_runtime_family", "")),
                    _fmt(float(row.get("NativeLikeAllowRate", 0.0) or 0.0)),
                    _fmt(float(row.get("MixedProdHarmRate", 0.0) or 0.0)),
                    _fmt(float(row.get("ProtectCriticalRate", 0.0) or 0.0)),
                    _fmt(float(row.get("BoringMasteryRate", 0.0) or 0.0)),
                    _fmt(float(row.get("LoopCompleteRate", 0.0) or 0.0)),
                ])
                + " |"
            )
    else:
        lines.extend([
            "## Stage 3: Runtime family validation",
            "",
            "No tasks reached runtime family validation because no task cleared",
            "exact render consistency under the current repository semantics.",
            "",
        ])

    lines.extend([
        "## Stage 4: Formal slice assembly",
        "",
        f"- Accepted tasks: `{int(formal_summary.get('accepted_count', 0))}`",
        f"- Rejected tasks: `{int(formal_summary.get('rejected_count', 0))}`",
        "",
        "A formal slice is assembled only from tasks that are exact-render valid",
        "and whose target family is actually present and dominant at runtime.",
    ])

    return "\n".join(lines) + "\n"

