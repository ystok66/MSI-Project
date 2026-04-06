"""
Encoders for RSA research project.

Provides color space conversion (RGB → Lab), normalization,
shape flattening, and scene encoding.

Output feature dimension: d = 12 (Lab:3 + shape:9)
No position encoding - region index only used for 4-dim output distribution.
"""

from typing import List, Tuple
import numpy as np

from world import Scene


# =============================================================================
# Color Space Conversion
# =============================================================================

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB (0-255) to CIE Lab color space.
    
    Uses the standard sRGB → XYZ → Lab conversion.
    Reference white: D65 illuminant.
    
    Args:
        rgb: Tuple of (R, G, B) values in range [0, 255]
        
    Returns:
        Tuple of (L, a, b) where:
            L: Lightness [0, 100]
            a: Green-red axis [-128, 127] approx
            b: Blue-yellow axis [-128, 127] approx
    """
    # Normalize RGB to [0, 1]
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # sRGB gamma correction (inverse companding)
    def linearize(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        else:
            return c / 12.92
    
    r, g, b = linearize(r), linearize(g), linearize(b)
    
    # RGB to XYZ (sRGB D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Reference white D65
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn
    
    # XYZ to Lab
    def f(t):
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        else:
            return t / (3 * delta ** 2) + 4.0 / 29.0
    
    fx, fy, fz = f(x), f(y), f(z)
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return (L, a, b_val)


def norm_lab(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Normalize Lab values to approximately [-1, 1] range.
    
    Transformation:
        L: [0, 100] → [-1, 1]
        a: [-128, 127] → [-1, 1]
        b: [-128, 127] → [-1, 1]
    
    Args:
        lab: Tuple of (L, a, b) in standard Lab range
        
    Returns:
        Tuple of normalized (L, a, b) in [-1, 1] range
    """
    L, a, b = lab
    
    L_norm = (L / 50.0) - 1.0          # [0,100] → [-1,1]
    a_norm = a / 128.0                  # [-128,127] → ~[-1,1]
    b_norm = b / 128.0                  # [-128,127] → ~[-1,1]
    
    return (L_norm, a_norm, b_norm)


# =============================================================================
# Shape Encoding
# =============================================================================

# Global setting for shape encoding mode
USE_GAUSSIAN_SHAPE = True  # Default: use Gaussian smoothing
GAUSSIAN_SIGMA = 0.5       # Sigma for Gaussian smoothing


def shape_to_vec(M: List[List[int]], use_gaussian: bool = None) -> List[float]:
    """
    Flatten a 3×3 shape matrix to a 9-dimensional vector.
    
    Row-major order (C-style): first row, then second row, then third row.
    
    Args:
        M: 3×3 matrix of 0/1 values
        use_gaussian: If True, apply Gaussian smoothing.
                      If None, use global USE_GAUSSIAN_SHAPE setting.
        
    Returns:
        List of 9 floats
    """
    if use_gaussian is None:
        use_gaussian = USE_GAUSSIAN_SHAPE
    
    if use_gaussian:
        return encode_shape_gaussian(M, sigma=GAUSSIAN_SIGMA)
    else:
        # Original hard 0/1 encoding
        result = []
        for row in M:
            for val in row:
                result.append(float(val))
        return result


def encode_shape_gaussian(occ_matrix: List[List[int]], sigma: float = 0.5) -> List[float]:
    """
    将 3x3 二值矩阵转换为带有空间相关性的软向量 (Gaussian Smoothing)。
    
    核心思想：
    - 每个 1 看作热源向周围辐射
    - 引入"空间邻域"概念，解决硬编码的稀疏性问题
    - 避免"零方差陷阱"，提升高斯概念模型的稳定性
    
    Args:
        occ_matrix: 原始 3x3 0/1 矩阵
        sigma: 控制模糊程度
               0.5 适合 3x3 网格 (只影响邻居)
               1.0 会太模糊 (变成一团)
               
    Returns:
        List of 9 floats in range [0.01, 0.99]
    """
    from scipy.ndimage import gaussian_filter
    
    # 1. 转换为 numpy 数组
    grid = np.array(occ_matrix, dtype=np.float64).reshape(3, 3)
    
    # 2. 高斯模糊 (核心: 引入空间相关性)
    # 如果 grid[0,0]=1, 那么 grid[0,1] 也会变成约 0.3~0.6
    # mode='constant' 意味着边界外视为 0
    soft_grid = gaussian_filter(grid, sigma=sigma, mode='constant', cval=0.0)
    
    # 3. 归一化 (保留峰值强度)
    # max=1 表示"这里确信有东西"
    # 不用 sum 归一化，因为实心物体理应比空心物体总能量大
    if soft_grid.max() > 0:
        soft_grid = soft_grid / soft_grid.max()
    
    # 4. 数值稳定性截断 (防止 Log(0) 或方差为 0)
    # 将数值限制在 [0.01, 0.99] 区间
    soft_grid = np.clip(soft_grid, 0.01, 0.99)
    
    return soft_grid.flatten().tolist()


# =============================================================================
# Scene Encoding
# =============================================================================

def encode_scene(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a scene into feature matrix and mask.
    
    For each region t ∈ {0,1,2,3}:
    - If non-empty: x_t = [norm_lab(color), flatten(shape)]  (12 dims)
    - If empty: x_t = zeros(12), mask[t] = False
    
    No position encoding - region index only used for output distribution.
    
    Args:
        scene: Scene with 4 regions
        
    Returns:
        X: np.ndarray of shape (4, 12) - feature matrix
        mask: np.ndarray of shape (4,) dtype bool - True if region is non-empty
    """
    d = 12  # 3 (Lab) + 9 (shape)
    X = np.zeros((4, d), dtype=np.float32)
    mask = np.zeros(4, dtype=bool)
    
    for t, obj in enumerate(scene.regions):
        if obj is None:
            # Empty region: X[t] stays zeros, mask[t] stays False
            continue
        
        # Encode color: RGB → Lab → normalized
        lab = rgb_to_lab(obj.color_rgb)
        lab_norm = norm_lab(lab)
        
        # Encode shape: 3×3 → 9-dim vector
        shape_vec = shape_to_vec(obj.occ.tolist())
        
        # Concatenate: [Lab(3), shape(9)]
        X[t, :3] = lab_norm
        X[t, 3:] = shape_vec
        
        mask[t] = True
    
    return X, mask
