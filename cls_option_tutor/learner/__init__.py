"""Active learner surface for the option-world tutor benchmark.

Legacy RSA support is intentionally not re-exported here; it remains available
through explicit module imports for archival and baseline experiments only.
"""

from .attention_model import AttentionModel
from .danger_head import DangerHead
from .episodic_memory import EpisodicMemory
from .learner_agent import LearnerAgent
from .policy import LearnerPolicy
from .semantic_scorer import DeterministicSemanticScorer

__all__ = [
    "AttentionModel",
    "DangerHead",
    "DeterministicSemanticScorer",
    "EpisodicMemory",
    "LearnerAgent",
    "LearnerPolicy",
    "SemanticScorer",
]

# Stable package-level alias for the active deterministic scorer.
SemanticScorer = DeterministicSemanticScorer
