"""
mlc_episode.py — Adapter converting MLC algebraic/mini-SCAN data to Episodes.
"""
from __future__ import annotations
import re
import os
from typing import Dict, List, Optional, Tuple
from cls_learner.interfaces import Example, Episode


def parse_algebraic_file(filepath: str) -> Episode:
    """
    Parse MLC algebraic task file into an Episode.

    Format:
        *SUPPORT*
        IN: word1 word2 OUT: COLOR1 COLOR2
        ...
        *QUERY*
        IN: word1 word2 OUT: COLOR1 COLOR2
        ...
        *GRAMMAR*
        ...
    """
    support, query = [], []
    current = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '*SUPPORT*':
                current = 'support'; continue
            elif line == '*QUERY*':
                current = 'query'; continue
            elif line == '*GRAMMAR*':
                current = 'grammar'; continue
            if current in ('support', 'query') and line:
                m = re.match(r'IN:\s+(.*?)\s+OUT:\s+(.*)', line)
                if m:
                    words = m.group(1).strip().split()
                    colors = m.group(2).strip().split()
                    ex = Example(words=words, output=colors)
                    if current == 'support':
                        support.append(ex)
                    else:
                        query.append(ex)
    return Episode(support=support, query=query)


def parse_miniscan_data(data_path: str) -> Tuple[List[Example], dict]:
    """
    Parse mini-SCAN human behavioral data.

    Returns (examples, human_data_dict).
    """
    # TODO: integrate with existing evaluation data parsing
    raise NotImplementedError("mini-SCAN parser to be migrated from eval scripts")


def dicts_to_examples(dicts: List[Dict]) -> List[Example]:
    """Convert list of {'input': [...], 'output': [...]} to Examples."""
    return [Example(words=d['input'], output=d['output']) for d in dicts]


def examples_to_dicts(examples: List[Example]) -> List[Dict]:
    """Convert Examples back to dict format for compatibility."""
    return [{'input': ex.words, 'output': ex.output} for ex in examples]
