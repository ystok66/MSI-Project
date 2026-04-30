#!/usr/bin/env python3
"""Lightweight validator for cls_family_benchmark_v1.
This checks file structure, support/query counts, and manifest consistency.
It does not replace runtime family validation in cls_option_tutor.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / 'tasks'
MANIFEST = ROOT / 'family_manifest.jsonl'

def parse_sections(path: Path):
    cur = None
    sections = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line in {'*SUPPORT*','*QUERY*','*GRAMMAR*'}:
            cur = line
            sections[cur] = []
        elif cur and line:
            sections[cur].append(line)
    return sections

def validate_task(path: Path):
    sec = parse_sections(path)
    for k in ['*SUPPORT*','*QUERY*','*GRAMMAR*']:
        assert k in sec, f'{path}: missing {k}'
    assert 12 <= len(sec['*SUPPORT*']) <= 16, f'{path}: bad support count'
    assert 8 <= len(sec['*QUERY*']) <= 12, f'{path}: bad query count'
    for line in sec['*SUPPORT*'] + sec['*QUERY*']:
        assert line.startswith('IN: ') and ' OUT: ' in line, f'{path}: bad IO line {line!r}'
        out = line.split(' OUT: ', 1)[1].split()
        assert out, f'{path}: empty OUT in {line!r}'
    for line in sec['*GRAMMAR*']:
        assert ' -> ' in line, f'{path}: bad grammar line {line!r}'
    return True

def main():
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(rows) == 40, f'expected 40 manifest rows, got {len(rows)}'
    for row in rows:
        p = ROOT / row['filename']
        assert p.exists(), f'missing task file {p}'
        validate_task(p)
    print(f'validated {len(rows)} tasks')

if __name__ == '__main__':
    main()
