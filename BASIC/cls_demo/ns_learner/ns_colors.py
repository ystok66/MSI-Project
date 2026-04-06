"""
ns_colors.py — Lab color space with noise injection.

Provides RGB→Lab conversion, a 6-color Lab palette,
Gaussian noise injection, and nearest-color quantization.

Used by the noise robustness experiment to compare
continuous (NIG/KL in Lab) vs discrete (Dirichlet over names).
"""
import numpy as np
from typing import Dict, Tuple


# ── RGB Palette ──────────────────────────────────────────────────

RGB_PALETTE: Dict[str, Tuple[int, int, int]] = {
    'BLUE':   (0,   0,   255),
    'RED':    (255, 0,   0),
    'GREEN':  (0,   128, 0),
    'YELLOW': (255, 255, 0),
    'PURPLE': (128, 0,   128),
    'PINK':   (255, 192, 203),
}


# ── sRGB → XYZ → CIELAB ─────────────────────────────────────────

def _srgb_to_linear(c: float) -> float:
    """sRGB gamma decode (0-1 range)."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _lab_f(t: float) -> float:
    """CIELAB nonlinear transform."""
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return t / (3 * delta ** 2) + 4.0 / 29.0


def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    """Convert sRGB (0-255) to CIELAB [L*, a*, b*].
    
    Uses D65 illuminant reference white.
    """
    # Normalize to 0-1 and linearize
    r, g, b = [_srgb_to_linear(c / 255.0) for c in rgb]
    
    # sRGB → XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    fx = _lab_f(x / xn)
    fy = _lab_f(y / yn)
    fz = _lab_f(z / zn)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    
    return np.array([L, a, b_val], dtype=np.float64)


# Pre-computed raw Lab palette (not normalized)
_RAW_LAB_PALETTE: Dict[str, np.ndarray] = {
    name: rgb_to_lab(rgb) for name, rgb in RGB_PALETTE.items()
}


# ── Lab normalization to [0, 1] ──────────────────────────────────
# Normalize each Lab dimension independently to [0,1] so NIG/KL
# mathematics work at the same scale as one-hot vectors.

_all_lab = np.array(list(_RAW_LAB_PALETTE.values()))
_LAB_MIN = _all_lab.min(axis=0) - 10.0   # 10-unit margin
_LAB_MAX = _all_lab.max(axis=0) + 10.0
_LAB_RANGE = _LAB_MAX - _LAB_MIN
_LAB_RANGE[_LAB_RANGE < 1e-6] = 1.0  # safety


def normalize_lab(raw_lab: np.ndarray) -> np.ndarray:
    """Normalize raw Lab vector to [0, 1] per dimension."""
    return (raw_lab - _LAB_MIN) / _LAB_RANGE


def denormalize_lab(norm_lab: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] Lab back to raw Lab."""
    return norm_lab * _LAB_RANGE + _LAB_MIN


# Normalized Lab palette (what the model sees)
LAB_PALETTE: Dict[str, np.ndarray] = {
    name: normalize_lab(raw) for name, raw in _RAW_LAB_PALETTE.items()
}

LAB_COLORS = list(LAB_PALETTE.keys())


def lab_vec(color_name: str) -> np.ndarray:
    """Get the normalized [0,1] Lab vector for a color name."""
    return LAB_PALETTE[color_name].copy()


# ── Noise injection ──────────────────────────────────────────────

def add_noise(vec: np.ndarray, sigma: float,
              rng: np.random.RandomState = None) -> np.ndarray:
    """Add Gaussian noise in normalized Lab space.
    
    sigma is in *raw* Lab units; we convert to normalized scale.
    """
    if sigma <= 0:
        return vec.copy()
    if rng is None:
        rng = np.random.RandomState()
    # Convert sigma from raw Lab to normalized units
    norm_sigma = sigma / _LAB_RANGE
    return vec + rng.normal(0, norm_sigma)


# ── Nearest color quantization ───────────────────────────────────

def nearest_color(vec: np.ndarray) -> str:
    """Find nearest palette color by L2 in normalized Lab space."""
    best_name = LAB_COLORS[0]
    best_dist = np.inf
    for name, ref in LAB_PALETTE.items():
        d = np.sum((vec - ref) ** 2)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def lab_vec_to_color(vec: np.ndarray) -> str:
    """Alias for nearest_color — drop-in for vec_to_color."""
    return nearest_color(vec)

# ── Lab → RGB (inverse) ─────────────────────────────────────────

def _lab_f_inv(t: float) -> float:
    """Inverse CIELAB nonlinear transform."""
    delta = 6.0 / 29.0
    if t > delta:
        return t ** 3
    return 3 * delta ** 2 * (t - 4.0 / 29.0)


def _linear_to_srgb(c: float) -> float:
    """sRGB gamma encode (0-1 range)."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def lab_to_rgb(lab: np.ndarray) -> Tuple[int, int, int]:
    """Convert CIELAB [L*, a*, b*] to sRGB (0-255).

    Uses D65 illuminant. Clips to valid gamut.
    """
    L, a, bv = lab[0], lab[1], lab[2]

    # Lab → XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - bv / 200

    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x = xn * _lab_f_inv(fx)
    y = yn * _lab_f_inv(fy)
    z = zn * _lab_f_inv(fz)

    # XYZ → linear sRGB
    rl =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # Gamma encode + clip
    r = int(np.clip(_linear_to_srgb(max(rl, 0)) * 255 + 0.5, 0, 255))
    g = int(np.clip(_linear_to_srgb(max(gl, 0)) * 255 + 0.5, 0, 255))
    b = int(np.clip(_linear_to_srgb(max(bl, 0)) * 255 + 0.5, 0, 255))
    return (r, g, b)


def norm_lab_to_mpl(norm_vec: np.ndarray) -> Tuple[float, float, float]:
    """Convert normalized [0,1] Lab vector to matplotlib RGB (0-1 floats).

    This is the key function for visualizing noisy Lab vectors.
    """
    raw = denormalize_lab(norm_vec)
    r, g, b = lab_to_rgb(raw)
    return (r / 255.0, g / 255.0, b / 255.0)


# ── Utility ──────────────────────────────────────────────────────

def lab_palette_mean() -> np.ndarray:
    """Mean normalized Lab vector across palette (for NIG mu0)."""
    return np.mean(list(LAB_PALETTE.values()), axis=0)


def lab_palette_info() -> Dict:
    """Return normalization info for display."""
    return {
        'min': _LAB_MIN, 'max': _LAB_MAX, 'range': _LAB_RANGE,
        'raw_palette': _RAW_LAB_PALETTE,
    }


if __name__ == '__main__':
    print("Lab Color Palette (normalized to [0,1]):")
    print(f"  {'Name':>8s}  {'RGB':>15s}  {'Raw L*':>7s}  {'Raw a*':>7s}  {'Raw b*':>7s}  │  {'nL':>5s}  {'na':>5s}  {'nb':>5s}")
    print(f"  {'-'*80}")
    for name in LAB_COLORS:
        rgb = RGB_PALETTE[name]
        raw = _RAW_LAB_PALETTE[name]
        norm = LAB_PALETTE[name]
        print(f"  {name:>8s}  {str(rgb):>15s}  {raw[0]:>7.1f}  {raw[1]:>7.1f}  {raw[2]:>7.1f}  │  {norm[0]:>.3f}  {norm[1]:>.3f}  {norm[2]:>.3f}")
    
    print(f"\n  Normalization: min={_LAB_MIN}, max={_LAB_MAX}")
    print(f"  Range: {_LAB_RANGE}")
    print(f"  Normalized mean: {lab_palette_mean()}")
    
    # Test noise + quantization at various σ
    print(f"\n  Noise quantization accuracy (1000 samples per color):")
    rng = np.random.RandomState(42)
    for sigma in [0, 2, 5, 10, 20, 30, 50]:
        correct = 0
        total = 0
        for name, ref in LAB_PALETTE.items():
            for _ in range(1000):
                noisy = add_noise(ref, sigma, rng)
                if nearest_color(noisy) == name:
                    correct += 1
                total += 1
        print(f"    σ={sigma:>3d}: {correct}/{total} ({correct*100/total:.1f}%)")
