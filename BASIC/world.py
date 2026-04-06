"""
World module for RSA research project.

Provides Scene/Obj dataclasses, scene generation, and utterance parsing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from templates import SHAPES, COLORS_RGB, SHAPE_NAMES, COLOR_NAMES


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Obj:
    """
    A single object in a scene region.
    
    Attributes:
        shape_name: Key into SHAPES dict
        color_rgb: RGB tuple (0-255)
        occ: 3×3 occupancy matrix (numpy array of 0/1)
    """
    shape_name: str
    color_rgb: Tuple[int, int, int]
    occ: np.ndarray  # shape (3, 3), dtype int, values 0 or 1


@dataclass
class Scene:
    """
    A scene with 4 regions (candidate slots).
    
    Each region can contain an object or be empty (None).
    The region index t ∈ {0,1,2,3} is only used for output distribution,
    not encoded into features.
    
    Attributes:
        regions: List of 4 elements, each Obj or None (empty)
    """
    regions: List[Optional[Obj]]  # len=4
    
    def __post_init__(self):
        assert len(self.regions) == 4, "Scene must have exactly 4 regions"


# =============================================================================
# Utterance Parsing
# =============================================================================

def parse_utterance(text: str) -> Tuple[int, List[str]]:
    """
    Parse an utterance string into (k, tokens).
    
    Rules:
    - If first field is an integer → k=that integer, rest are tokens
    - Otherwise k=1, all fields are tokens
    - All tokens are lowercased
    - Empty input returns (1, [])
    
    Examples:
        "1 blue box"  → (1, ["blue", "box"])
        "blue box"    → (1, ["blue", "box"])
        "2 TV green"  → (2, ["tv", "green"])
        ""            → (1, [])
    
    Args:
        text: Input utterance string
        
    Returns:
        (k, tokens): cardinality and list of lowercase tokens
    """
    text = text.strip()
    if not text:
        return (1, [])
    
    parts = text.split()
    
    # Check if first part is an integer
    try:
        k = int(parts[0])
        tokens = [t.lower() for t in parts[1:]]
    except ValueError:
        k = 1
        tokens = [t.lower() for t in parts]
    
    return (k, tokens)


# =============================================================================
# Scene Generation
# =============================================================================

def sample_scene(
    rng: np.random.Generator,
    p_empty: float = 0.3,
    shapes: Optional[List[str]] = None,
    colors: Optional[List[str]] = None
) -> Scene:
    """
    Sample a random scene with 4 regions.
    
    Each region independently:
    - With probability p_empty: empty (None)
    - Otherwise: random shape and color
    
    Args:
        rng: NumPy random generator for reproducibility
        p_empty: Probability that each region is empty (default 0.3)
        shapes: List of shape names to sample from (default: all SHAPE_NAMES)
        colors: List of color names to sample from (default: all COLOR_NAMES)
        
    Returns:
        Scene with 4 regions
    """
    if shapes is None:
        shapes = SHAPE_NAMES
    if colors is None:
        colors = COLOR_NAMES
    
    regions: List[Optional[Obj]] = []
    
    for _ in range(4):
        if rng.random() < p_empty:
            regions.append(None)
        else:
            # Sample shape
            shape_name = rng.choice(shapes)
            shape_matrix = SHAPES[shape_name]
            occ = np.array(shape_matrix, dtype=np.int32)
            
            # Sample color
            color_name = rng.choice(colors)
            color_rgb = COLORS_RGB[color_name]
            
            regions.append(Obj(
                shape_name=shape_name,
                color_rgb=color_rgb,
                occ=occ
            ))
    
    return Scene(regions=regions)
