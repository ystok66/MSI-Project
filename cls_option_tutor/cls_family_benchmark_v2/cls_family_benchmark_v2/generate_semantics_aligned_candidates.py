#!/usr/bin/env python3
"""Generate a semantics-aligned family benchmark candidate pool.

Method:
1. start from source tasks that already run under the current repository
2. choose source tasks screened for mixed/protect family potential
3. generate semantic-equivalent clones by consistently renaming lowercase
   vocabulary only
4. keep outputs exactly aligned to the real renderer semantics
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
SOURCE_DATA = REPO_ROOT / "BASIC" / "cls_learner" / "data"
TASK_DIR = ROOT / "tasks"
MANIFEST_CSV = ROOT / "family_manifest.csv"
MANIFEST_JSONL = ROOT / "family_manifest.jsonl"


MIXED_SOURCES: List[Tuple[str, int, float]] = [
    ("000016", 3, 0.7600),
    ("000006", 3, 0.6129),
    ("000013", 3, 0.4643),
    ("000004", 3, 0.4231),
    ("000020", 3, 0.3200),
    ("000003", 3, 0.3103),
]

PROTECT_SOURCES: List[Tuple[str, int, float]] = [
    ("000001", 12, 0.0769),
]


def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = None
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


def _unique_lower_tokens(sections: Dict[str, List[str]]) -> List[str]:
    keep = {"u1", "u2", "x1", "x2"}
    tokens: List[str] = []

    def maybe_add(tok: str) -> None:
        if tok in keep:
            return
        if tok.startswith("[") and tok.endswith("]"):
            return
        if tok.isupper():
            return
        if not re.fullmatch(r"[a-z][a-z0-9_]*", tok):
            return
        if tok not in tokens:
            tokens.append(tok)

    for line in sections.get("SUPPORT", []) + sections.get("QUERY", []):
        left = line.split(" OUT: ", 1)[0]
        inp = left.replace("IN: ", "", 1).strip().split()
        for tok in inp:
            maybe_add(tok)

    for line in sections.get("GRAMMAR", []):
        if "->" not in line:
            continue
        lhs, _ = line.split("->", 1)
        for tok in lhs.strip().split():
            maybe_add(tok)

    return tokens


def _make_word_pool(seed: int, count: int) -> List[str]:
    rng = random.Random(seed)
    starts = [
        "ba", "be", "bi", "bo", "bu",
        "ca", "ce", "ci", "co", "cu",
        "da", "de", "di", "do", "du",
        "fa", "fe", "fi", "fo", "fu",
        "ga", "ge", "gi", "go", "gu",
        "ka", "ke", "ki", "ko", "ku",
        "la", "le", "li", "lo", "lu",
        "ma", "me", "mi", "mo", "mu",
        "na", "ne", "ni", "no", "nu",
        "pa", "pe", "pi", "po", "pu",
        "ra", "re", "ri", "ro", "ru",
        "sa", "se", "si", "so", "su",
        "ta", "te", "ti", "to", "tu",
        "va", "ve", "vi", "vo", "vu",
        "za", "ze", "zi", "zo", "zu",
    ]
    ends = [
        "lan", "vek", "mor", "tin", "ruk", "pas", "lod", "mir", "sen",
        "fal", "dor", "nim", "gar", "pel", "vos", "kim", "zan", "rel",
        "tas", "gor", "vel", "mun", "rad", "pol", "nex", "sor", "bik",
    ]

    seen = set()
    result = []
    while len(result) < count:
        token = rng.choice(starts) + rng.choice(ends)
        if token not in seen and token not in {"u1", "u2", "x1", "x2"}:
            seen.add(token)
            result.append(token)
    return result


def _rename_line_io(line: str, mapping: Dict[str, str]) -> str:
    left, right = line.split(" OUT: ", 1)
    inp = left.replace("IN: ", "", 1).strip().split()
    renamed = [mapping.get(tok, tok) for tok in inp]
    return f"IN: {' '.join(renamed)} OUT: {right.strip()}"


def _rename_line_grammar(line: str, mapping: Dict[str, str]) -> str:
    lhs, rhs = line.split("->", 1)
    lhs_tokens = lhs.strip().split()
    renamed_lhs = [mapping.get(tok, tok) for tok in lhs_tokens]
    return f"{' '.join(renamed_lhs)} -> {rhs.strip()}"


def _render_task_text(sections: Dict[str, List[str]], mapping: Dict[str, str]) -> str:
    lines: List[str] = ["*SUPPORT*"]
    lines.extend(_rename_line_io(line, mapping) for line in sections["SUPPORT"])
    lines.append("")
    lines.append("*QUERY*")
    lines.extend(_rename_line_io(line, mapping) for line in sections["QUERY"])
    lines.append("")
    lines.append("*GRAMMAR*")
    lines.extend(_rename_line_grammar(line, mapping) for line in sections["GRAMMAR"])
    return "\n".join(lines).strip() + "\n"


def _make_manifest_row(
    *,
    task_id: str,
    filename: str,
    intended_family: str,
    generator_mode: str,
    source_task_id: str,
    source_family_rate: float,
    variant_index: int,
    support_count: int,
    query_count: int,
    n_nouns: int,
    n_operators: int,
) -> dict:
    return {
        "task_id": task_id,
        "filename": filename,
        "intended_family": intended_family,
        "generator_mode": generator_mode,
        "generation_method": "screened_source_clone_semantics_aligned",
        "source_task_id": source_task_id,
        "source_family_rate_estimate": f"{source_family_rate:.4f}",
        "variant_index": variant_index,
        "difficulty": "screened_clone",
        "n_nouns": n_nouns,
        "n_operators": n_operators,
        "support_count": support_count,
        "query_count": query_count,
        "needs_runtime_family_validation": True,
        "static_validator_version": "exact_render_v1",
    }


def _collect_rows_for_family(
    *,
    family_prefix: str,
    intended_family: str,
    generator_mode: str,
    source_specs: Iterable[Tuple[str, int, float]],
    task_index_start: int,
) -> List[dict]:
    rows: List[dict] = []
    next_idx = task_index_start
    for source_task_id, n_variants, source_rate in source_specs:
        source_path = SOURCE_DATA / f"{source_task_id}.txt"
        text = source_path.read_text(encoding="utf-8")
        sections = _parse_sections(text)
        source_tokens = _unique_lower_tokens(sections)
        n_nouns = sum(
            1
            for line in sections["GRAMMAR"]
            if "->" in line
            and len(line.split("->", 1)[0].strip().split()) == 1
            and line.split("->", 1)[1].strip().isupper()
        )
        n_operators = len(sections["GRAMMAR"]) - n_nouns

        for variant_i in range(1, n_variants + 1):
            task_id = f"{family_prefix}_{next_idx:03d}"
            filename = f"tasks/{task_id}.txt"
            pool = _make_word_pool(seed=hash((family_prefix, source_task_id, variant_i)) & 0xFFFFFFFF, count=len(source_tokens))
            mapping = {src: dst for src, dst in zip(source_tokens, pool)}
            rendered_text = _render_task_text(sections, mapping)
            (TASK_DIR / f"{task_id}.txt").write_text(rendered_text, encoding="utf-8")

            rows.append(
                _make_manifest_row(
                    task_id=task_id,
                    filename=filename,
                    intended_family=intended_family,
                    generator_mode=generator_mode,
                    source_task_id=source_task_id,
                    source_family_rate=source_rate,
                    variant_index=variant_i,
                    support_count=len(sections["SUPPORT"]),
                    query_count=len(sections["QUERY"]),
                    n_nouns=n_nouns,
                    n_operators=n_operators,
                )
            )
            next_idx += 1
    return rows


def main() -> None:
    if TASK_DIR.exists():
        shutil.rmtree(TASK_DIR)
    TASK_DIR.mkdir(parents=True, exist_ok=True)

    mixed_rows = _collect_rows_for_family(
        family_prefix="mixed_v2",
        intended_family="MIXED_PROD_HARM_HEAVY",
        generator_mode="diagnostic_quota_mixed_prod_harm_heavy",
        source_specs=MIXED_SOURCES,
        task_index_start=1,
    )
    protect_rows = _collect_rows_for_family(
        family_prefix="protect_v2",
        intended_family="PROTECT_CRITICAL_HEAVY",
        generator_mode="diagnostic_quota_protect_critical_heavy",
        source_specs=PROTECT_SOURCES,
        task_index_start=1,
    )
    rows = mixed_rows + protect_rows

    fieldnames = list(rows[0].keys()) if rows else []
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with MANIFEST_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} tasks")
    print(f"[ok] mixed={len(mixed_rows)} protect={len(protect_rows)}")
    print(f"[ok] manifest={MANIFEST_CSV}")


if __name__ == "__main__":
    main()

