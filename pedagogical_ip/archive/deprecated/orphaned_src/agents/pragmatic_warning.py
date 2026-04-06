"""
PragmaticWarner — shared protocol for warning systems.

Inspired by pypragmods' L0/S1/L1 factoring:
  - select_utterance() ≈ S1 (speaker chooses best utterance)
  - listener_update()  ≈ L0 (listener updates belief from utterance)

Both RSAWarner (v0-v1d) and LaneWarner (V2) implement this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PragmaticWarner(Protocol):
    """Minimal pragmatic warning interface.

    Both warning systems (RSA region-based and V2 lane-based) follow the
    same speaker–listener pattern. This protocol captures the shared
    structure without forcing implementation details.

    Typical flow:
        utterance = warner.select_utterance(state_info)
        effect    = warner.listener_update(utterance, belief, **kwargs)
    """

    def select_utterance(self, state_info: dict) -> str | None:
        """S1: speaker selects the best utterance given current state.

        Args:
            state_info: dict with system-specific context, e.g.:
              - RSA: learner_belief_risk_mean, true_risk, agent_pos, ...
              - V2:  candidate_cells, feature_belief, risk_head, ...

        Returns:
            Utterance string, or None if nothing worth saying.
        """
        ...

    def listener_update(self, utterance: str, belief: Any, **kwargs) -> Any:
        """L0: listener updates belief given an utterance.

        Args:
            utterance: the selected utterance string
            belief: the listener's belief state (BeliefMap or FeatureBeliefMap)
            **kwargs: system-specific parameters

        Returns:
            Effect record (system-specific dataclass or dict).
        """
        ...
