"""
episodic_memory.py — Within-block reveal-history integration.

Implements §8 reveal update:
    After wrong choice → store (ν_chosen, Y_revealed, d) in memory.
    Memory biases future semantic discrimination within the block.

Key design: memory is per-BLOCK, not per-query.
Cross-block evaluation should NOT leak memory.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..interfaces import RevealEvent


@dataclass
class EpisodicMemory:
    """Within-block episodic memory from reveal events.

    Stores revealed wrong-option outputs to help the learner:
    1. Avoid re-selecting options similar to known-wrong ones
    2. Improve danger estimation from observed (v, d) pairs
    3. Narrow semantic hypotheses by elimination

    §6.2 / §8: active during the block, frozen across blocks.
    """
    # Indexed by option text (as tuple for hashability)
    reveals: Dict[Tuple[str, ...], RevealEvent] = field(default_factory=dict)

    # Quick lookup sets
    known_wrong_outputs: Set[Tuple[str, ...]] = field(default_factory=set)
    known_wrong_texts: Set[Tuple[str, ...]] = field(default_factory=set)

    # Danger observations for predictor training: (v, d) pairs
    danger_observations: List[Tuple] = field(default_factory=list)

    def write_reveal(self, event: RevealEvent) -> None:
        """Store a reveal event after a wrong choice."""
        key = tuple(event.option_text)
        self.reveals[key] = event
        self.known_wrong_texts.add(key)
        if event.revealed_output:
            self.known_wrong_outputs.add(tuple(event.revealed_output))
        self.danger_observations.append(
            (event.danger_vec.copy(), event.damage))

    def is_known_wrong(self, text: List[str]) -> bool:
        """Check if an option text was already revealed as wrong."""
        return tuple(text) in self.known_wrong_texts

    def similarity_to_known_wrong(self, output: List[str]) -> float:
        """How similar is this output to known-wrong outputs?

        Returns fraction of known-wrong outputs that match exactly.
        Useful for elimination-based scoring.
        """
        if not self.known_wrong_outputs:
            return 0.0
        key = tuple(output)
        return 1.0 if key in self.known_wrong_outputs else 0.0

    def get_elimination_penalty(self, text: List[str],
                                rendered: Optional[List[str]]) -> float:
        """Penalty for choosing an option similar to known-wrong ones.

        Returns a negative score (0 = no penalty, -1 = definitely wrong).
        """
        if self.is_known_wrong(text):
            return -10.0  # hard penalty: exact same text already tried

        if rendered is not None and tuple(rendered) in self.known_wrong_outputs:
            return -5.0   # same rendered output as a known-wrong

        return 0.0

    @property
    def n_reveals(self) -> int:
        return len(self.reveals)

    def reset(self) -> None:
        """Clear memory (new block)."""
        self.reveals.clear()
        self.known_wrong_outputs.clear()
        self.known_wrong_texts.clear()
        self.danger_observations.clear()
