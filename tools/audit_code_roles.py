"""
audit_code_roles.py - Static import-graph and unreferenced-file audit for cls_option_tutor.

Default outputs:
    docs/import_graph_report.md
    docs/unreferenced_files_report.md

Scope:
    - scans Python files under a package root (default: cls_option_tutor)
    - builds a local import graph using AST
    - reports local imports, reverse imports, and reachability from entrypoints
    - highlights files with no local importers and files not reachable from main entrypoints

This script is intentionally conservative:
    - it only does static analysis
    - it does not move or delete files
    - it does not try to infer runtime-only imports
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class FileRecord:
    path: Path
    relpath: Path
    module: str
    kind: str


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def module_name_for_file(repo_root: Path, py_path: Path) -> str:
    rel = py_path.relative_to(repo_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def file_kind(relpath: Path, package_root: str) -> str:
    parts = relpath.parts
    if relpath == Path(package_root) / "experiments" / "run_learning_increment_micro.py":
        return "main_entrypoint"
    if relpath == Path(package_root) / "exp_option_level.py":
        return "legacy_runner"
    if "tests" in parts:
        return "test"
    if "results" in parts:
        return "results_script"
    if "tmp" in parts:
        return "tmp"
    if relpath.name == "__init__.py":
        return "package_init"
    return "module"


def discover_files(repo_root: Path, package_root: str) -> Dict[str, FileRecord]:
    package_dir = repo_root / package_root
    records: Dict[str, FileRecord] = {}
    for py_path in sorted(package_dir.rglob("*.py")):
        relpath = py_path.relative_to(repo_root)
        module = module_name_for_file(repo_root, py_path)
        records[module] = FileRecord(
            path=py_path,
            relpath=relpath,
            module=module,
            kind=file_kind(relpath, package_root),
        )
    return records


def current_package_for_resolution(module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def longest_local_module(name: str, local_modules: Set[str]) -> Optional[str]:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in local_modules:
            return cand
    return None


def resolve_import_from(
    base_package: str,
    node: ast.ImportFrom,
) -> Optional[str]:
    if node.level and node.level > 0:
        rel = "." * node.level + (node.module or "")
        try:
            return importlib.util.resolve_name(rel, base_package)
        except Exception:
            return None
    return node.module


def local_imports_for_file(
    record: FileRecord,
    local_modules: Set[str],
) -> Tuple[Set[str], List[str]]:
    imported: Set[str] = set()
    parse_errors: List[str] = []
    try:
        tree = ast.parse(record.path.read_text(encoding="utf-8"), filename=str(record.path))
    except Exception as exc:
        parse_errors.append(f"parse_error: {exc}")
        return imported, parse_errors

    base_package = current_package_for_resolution(record.module, record.path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                match = longest_local_module(alias.name, local_modules)
                if match:
                    imported.add(match)
        elif isinstance(node, ast.ImportFrom):
            resolved = resolve_import_from(base_package, node)
            if not resolved:
                continue

            matched_any = False
            for alias in node.names:
                if alias.name == "*":
                    if resolved in local_modules:
                        imported.add(resolved)
                        matched_any = True
                    continue

                subcand = f"{resolved}.{alias.name}"
                if subcand in local_modules:
                    imported.add(subcand)
                    matched_any = True
                elif resolved in local_modules:
                    imported.add(resolved)
                    matched_any = True

            if not matched_any and resolved in local_modules:
                imported.add(resolved)

    imported.discard(record.module)
    return imported, parse_errors


def reachable_from(start_modules: Iterable[str], graph: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    q = deque(start_modules)
    while q:
        mod = q.popleft()
        if mod in seen:
            continue
        seen.add(mod)
        for nxt in graph.get(mod, set()):
            if nxt not in seen:
                q.append(nxt)
    return seen


def fmt_module_list(modules: Iterable[str]) -> str:
    items = sorted(set(modules))
    if not items:
        return "-"
    return ", ".join(f"`{m}`" for m in items)


def write_import_graph_report(
    out_path: Path,
    records: Dict[str, FileRecord],
    imports_by_file: Dict[str, Set[str]],
    importers_by_file: Dict[str, Set[str]],
    parse_errors_by_file: Dict[str, List[str]],
    reachable_main: Set[str],
    reachable_tests: Set[str],
    main_entrypoints: List[str],
    test_entrypoints: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Import Graph Report")
    lines.append("")
    lines.append("Generated by `tools/audit_code_roles.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total Python modules scanned: `{len(records)}`")
    lines.append(f"- Main entrypoints: {fmt_module_list(main_entrypoints)}")
    lines.append(f"- Test entrypoints: {fmt_module_list(test_entrypoints)}")
    lines.append(f"- Reachable from main entrypoints: `{len(reachable_main)}`")
    lines.append(f"- Reachable from tests: `{len(reachable_tests)}`")
    lines.append("")
    lines.append("## File Table")
    lines.append("")
    lines.append("| File | Kind | Imported By | Imports | Reachable Main | Reachable Tests | Parse |")
    lines.append("|---|---|---:|---:|---|---|---|")

    def sort_key(item: FileRecord) -> Tuple[str, str]:
        return (item.kind, item.relpath.as_posix())

    for rec in sorted(records.values(), key=sort_key):
        imported_by_n = len(importers_by_file.get(rec.module, set()))
        imports_n = len(imports_by_file.get(rec.module, set()))
        parse_state = "ok" if not parse_errors_by_file.get(rec.module) else "error"
        lines.append(
            f"| [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()}) "
            f"| `{rec.kind}` | `{imported_by_n}` | `{imports_n}` "
            f"| `{'yes' if rec.module in reachable_main else 'no'}` "
            f"| `{'yes' if rec.module in reachable_tests else 'no'}` "
            f"| `{parse_state}` |"
        )

    lines.append("")
    lines.append("## Detailed Imports")
    lines.append("")

    for rec in sorted(records.values(), key=sort_key):
        lines.append(f"### `{rec.module}`")
        lines.append("")
        lines.append(f"- File: [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()})")
        lines.append(f"- Kind: `{rec.kind}`")
        lines.append(f"- Imports: {fmt_module_list(imports_by_file.get(rec.module, set()))}")
        lines.append(f"- Imported by: {fmt_module_list(importers_by_file.get(rec.module, set()))}")
        lines.append(f"- Reachable from main entrypoints: `{'yes' if rec.module in reachable_main else 'no'}`")
        lines.append(f"- Reachable from tests: `{'yes' if rec.module in reachable_tests else 'no'}`")
        if parse_errors_by_file.get(rec.module):
            for err in parse_errors_by_file[rec.module]:
                lines.append(f"- Parse issue: `{err}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_unreferenced_report(
    out_path: Path,
    records: Dict[str, FileRecord],
    imports_by_file: Dict[str, Set[str]],
    importers_by_file: Dict[str, Set[str]],
    reachable_main: Set[str],
    reachable_tests: Set[str],
    main_entrypoints: List[str],
) -> None:
    standalone_kinds = {"main_entrypoint", "legacy_runner", "test", "results_script", "tmp"}

    no_local_importers = [
        rec for rec in records.values()
        if not importers_by_file.get(rec.module)
        and rec.kind not in standalone_kinds
    ]
    not_reachable_from_main = [
        rec for rec in records.values()
        if rec.module not in reachable_main
        and rec.kind not in {"test", "results_script", "tmp"}
    ]
    test_only_reachable = [
        rec for rec in records.values()
        if rec.module in reachable_tests
        and rec.module not in reachable_main
        and rec.kind not in {"test", "results_script", "tmp"}
    ]
    standalone_scripts = [
        rec for rec in records.values()
        if rec.kind in {"results_script", "tmp"}
    ]

    lines: List[str] = []
    lines.append("# Unreferenced Files Report")
    lines.append("")
    lines.append("Generated by `tools/audit_code_roles.py`.")
    lines.append("")
    lines.append("This report is conservative:")
    lines.append("")
    lines.append("- `no local importers` does not automatically mean safe to archive")
    lines.append("- main entrypoints, tests, and result scripts are treated separately")
    lines.append("- runtime-only imports are not captured by static analysis")
    lines.append("")

    lines.append("## Main entrypoints used for reachability")
    lines.append("")
    lines.append(f"{fmt_module_list(main_entrypoints)}")
    lines.append("")

    lines.append("## No Local Importers")
    lines.append("")
    if not no_local_importers:
        lines.append("- None")
    else:
        for rec in sorted(no_local_importers, key=lambda r: r.relpath.as_posix()):
            lines.append(
                f"- [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()}) "
                f"`{rec.kind}`"
            )
    lines.append("")

    lines.append("## Not Reachable From Main Entrypoints")
    lines.append("")
    if not not_reachable_from_main:
        lines.append("- None")
    else:
        for rec in sorted(not_reachable_from_main, key=lambda r: r.relpath.as_posix()):
            lines.append(
                f"- [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()}) "
                f"`{rec.kind}`"
            )
    lines.append("")

    lines.append("## Reachable Only From Tests")
    lines.append("")
    if not test_only_reachable:
        lines.append("- None")
    else:
        for rec in sorted(test_only_reachable, key=lambda r: r.relpath.as_posix()):
            lines.append(
                f"- [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()}) "
                f"`{rec.kind}`"
            )
    lines.append("")

    lines.append("## Standalone Result / Temp Scripts")
    lines.append("")
    if not standalone_scripts:
        lines.append("- None")
    else:
        for rec in sorted(standalone_scripts, key=lambda r: r.relpath.as_posix()):
            lines.append(
                f"- [{rec.relpath.as_posix()}](/F:/SCAI/Learning-agent/{rec.relpath.as_posix()}) "
                f"`{rec.kind}`"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local import graph and unreferenced files.")
    parser.add_argument("--package-root", default="cls_option_tutor")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--import-graph-report", default="import_graph_report.md")
    parser.add_argument("--unreferenced-report", default="unreferenced_files_report.md")
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    docs_dir = repo_root / args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    records = discover_files(repo_root, args.package_root)
    local_modules = set(records.keys())

    imports_by_file: Dict[str, Set[str]] = {}
    importers_by_file: Dict[str, Set[str]] = defaultdict(set)
    parse_errors_by_file: Dict[str, List[str]] = {}

    for module, rec in records.items():
        imported, parse_errors = local_imports_for_file(rec, local_modules)
        imports_by_file[module] = imported
        if parse_errors:
            parse_errors_by_file[module] = parse_errors
        for dep in imported:
            importers_by_file[dep].add(module)

    main_entrypoints = [
        rec.module for rec in records.values() if rec.kind in {"main_entrypoint", "legacy_runner"}
    ]
    test_entrypoints = [
        rec.module for rec in records.values() if rec.kind == "test"
    ]

    reachable_main = reachable_from(main_entrypoints, imports_by_file)
    reachable_tests = reachable_from(test_entrypoints, imports_by_file)

    write_import_graph_report(
        docs_dir / args.import_graph_report,
        records,
        imports_by_file,
        importers_by_file,
        parse_errors_by_file,
        reachable_main,
        reachable_tests,
        main_entrypoints,
        test_entrypoints,
    )
    write_unreferenced_report(
        docs_dir / args.unreferenced_report,
        records,
        imports_by_file,
        importers_by_file,
        reachable_main,
        reachable_tests,
        main_entrypoints,
    )

    print(f"Wrote {docs_dir / args.import_graph_report}")
    print(f"Wrote {docs_dir / args.unreferenced_report}")
    print(f"Scanned {len(records)} modules under {args.package_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
