"""
Intervention types and data structures for the teacher/robot.

Phase 8: unified intervention family.
All intervention actions share one schema. BLOCK_PATH is legacy/debug only
and must NOT enter the main intervention comparison.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class InterventionType(enum.Enum):
    """Available robot intervention types."""
    WAIT = "WAIT"
    WARN = "WARN"
    UNLOCK_DOOR = "UNLOCK_DOOR"
    DROP_SHIELD = "DROP_SHIELD"
    BLOCK_PATH = "BLOCK_PATH"       # legacy/debug only — NOT in main comparison


# Main intervention family (used for unified counterfactual comparison)
MAIN_INTERVENTION_FAMILY = frozenset({
    InterventionType.WAIT,
    InterventionType.WARN,
    InterventionType.UNLOCK_DOOR,
    InterventionType.DROP_SHIELD,
})


class ItemType(enum.Enum):
    """Supported item types. Phase 8: shield only."""
    SHIELD = "shield"


# Single source of truth for shield risk reduction
SHIELD_DEFAULT_RISK_REDUCTION = 0.5

VALID_ITEM_LOCATIONS = frozenset({"current_cell"})


@dataclass
class ItemEffect:
    """Effect of an item on the agent's traversal.

    Semantics must match identically in planner, predictor, and runner.
    """
    item_type: ItemType
    risk_reduction: float = SHIELD_DEFAULT_RISK_REDUCTION
    auto_consume: bool = True
    location: str = "current_cell"


@dataclass
class InventoryState:
    """Binary inventory: agent holds 0 or 1 shield.

    No stacking. No multi-item. Consumed on first risky traversal.
    """
    shield: int = 0                    # 0 or 1
    shield_risk_reduction: float = SHIELD_DEFAULT_RISK_REDUCTION

    def has_shield(self) -> bool:
        return self.shield > 0

    def consume_shield(self) -> bool:
        """Consume one shield. Returns True if consumed."""
        if self.shield > 0:
            self.shield = 0
            return True
        return False

    def add_shield(self) -> bool:
        """Add shield. Returns False if already full (no stacking)."""
        if self.shield >= 1:
            return False
        self.shield = 1
        return True

    def clone(self) -> InventoryState:
        """Clone for counterfactual rollout (read-only)."""
        return InventoryState(
            shield=self.shield,
            shield_risk_reduction=self.shield_risk_reduction,
        )


# Warning vocabulary — v1a (includes both v0 aliases and RSA utterances)
WARNING_VOCAB = [
    # v1a RSA utterances
    "LEFT_RISKY",
    "RIGHT_RISKY",
    "UPPER_RISKY",
    "LOWER_RISKY",
    "DOOR_PATH_SAFE",
    "CURRENT_PATH_RISKY",
    # v0 legacy aliases
    "LEFT_AREA_RISKY",
    "RIGHT_AREA_RISKY",
    "CURRENT_PLAN_RISKY",
]


@dataclass
class Intervention:
    """A teacher/robot intervention action."""
    type: InterventionType
    param: str = ""       # e.g. warning message, door_id
    duration: int = 0     # for DROP_SHIELD

    @staticmethod
    def wait() -> Intervention:
        return Intervention(type=InterventionType.WAIT)

    @staticmethod
    def warn(message: str) -> Intervention:
        assert message in WARNING_VOCAB, f"Unknown warning: {message}"
        return Intervention(type=InterventionType.WARN, param=message)

    @staticmethod
    def unlock_door(door_id: str = "0") -> Intervention:
        return Intervention(type=InterventionType.UNLOCK_DOOR, param=door_id)

    @staticmethod
    def item_drop(
        item_type: str = "shield",
        location: str = "current_cell",
    ) -> Intervention:
        """Drop an item at specified location.

        Phase 8 scope: only shield, only current_cell.
        """
        if item_type not in {it.value for it in ItemType}:
            raise ValueError(f"Unknown item_type: {item_type}. "
                             f"Valid: {[it.value for it in ItemType]}")
        if location not in VALID_ITEM_LOCATIONS:
            raise ValueError(f"Invalid location: {location}. "
                             f"Valid: {list(VALID_ITEM_LOCATIONS)}")
        return Intervention(
            type=InterventionType.DROP_SHIELD,
            param=f"{item_type}@{location}",
        )

    @staticmethod
    def drop_shield(duration: int = 5) -> Intervention:
        """Legacy factory — delegates to item_drop."""
        return Intervention.item_drop(item_type="shield", location="current_cell")

    @staticmethod
    def block_path(duration: int = 3) -> Intervention:
        """Pedagogical blocking — legacy/debug only.

        NOT part of main intervention family comparison.
        """
        return Intervention(type=InterventionType.BLOCK_PATH, duration=duration)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        d: dict = {"type": self.type.value}
        if self.param:
            d["param"] = self.param
        if self.duration:
            d["duration"] = self.duration
        return d
