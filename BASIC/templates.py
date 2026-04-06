"""
Shape templates and color palette for RSA research project.

Shape templates are 3×3 binary occupancy matrices.
Colors are defined as RGB tuples for rendering purposes.
Note: Render color names are separate from language token vocabulary.
"""

from typing import Dict, List, Tuple

# =============================================================================
# Shape Templates (3×3 binary occupancy)
# =============================================================================
# Each shape is a 3×3 grid where 1=occupied, 0=empty
# Stored as list[list[int]] for easy human readability

SHAPES: Dict[str, List[List[int]]] = {
    # Basic shapes
    "box": [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    "solid": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    
    # Bars
    "hbar": [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ],
    "vbar": [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ],
    
    # L-shapes (4 rotations)
    "l": [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ],
    "l_90": [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ],
    "l_180": [
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ],
    "l_270": [
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    
    # T-shapes
    "t": [
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ],
    "t_90": [
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0]
    ],
    "t_180": [
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ],
    "t_270": [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1]
    ],
    
    # Cross / Plus
    "cross": [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ],
    
    # S-shapes (2 rotations)
    "s": [
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ],
    "s_90": [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ],
    
    # Z-shapes (2 rotations)
    "z": [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ],
    "z_90": [
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ],
    
    # Corners (4 rotations)
    "corner": [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ],
    "corner_90": [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],
    "corner_180": [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ],
    "corner_270": [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ],
    
    # Diagonal
    "diag": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ],
    "diag_90": [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ],
    
    # Dot / Single pixel
    "dot": [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],
    
    # Donut / Ring (same as box but explicit name)
    "donut": [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    
    # U-shapes
    "u": [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    "u_90": [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    "u_180": [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]
    ],
    "u_270": [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ],
}

# =============================================================================
# Color Palette (for rendering)
# =============================================================================
# RGB values (0-255) for each render color
# These are independent of language tokens

COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "red":     (220, 30, 30),
    "green":   (30, 200, 60),
    "blue":    (50, 80, 220),
    "yellow":  (240, 220, 60),
    "orange":  (255, 140, 40),
    "purple":  (150, 50, 200),
    "cyan":    (40, 200, 220),
    "magenta": (220, 50, 180),
    "brown":   (140, 90, 50),
    "pink":    (255, 180, 200),
    "gray":    (130, 130, 130),
    "white":   (245, 245, 245),
}

# =============================================================================
# Helper lists for sampling
# =============================================================================
SHAPE_NAMES: List[str] = list(SHAPES.keys())
COLOR_NAMES: List[str] = list(COLORS_RGB.keys())
