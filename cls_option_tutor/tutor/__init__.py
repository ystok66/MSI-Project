"""Active tutor surface for the option-world tutor benchmark.

Exports the current sparse-tutor mainline plus benchmark baselines that are
still used in comparisons. Legacy research modules remain available by explicit
module import but are not promoted through ``__all__``.
"""

from .direct_answer_tutor import DirectAnswerTutor
from .scripted_protocols import ScriptedProtocolRunner
from .sparse_tutor import SparseTutorAgent

__all__ = [
    "DirectAnswerTutor",
    "ScriptedProtocolRunner",
    "SparseTutorAgent",
]
