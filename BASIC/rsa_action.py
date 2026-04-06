"""
RSA Physics Engine - Neuro-symbolic World Model

核心思想：
- 将物理规则编码为联合高斯概念 [State(36D), Delta(36D)]
- State: 当前 3x3 RGBI 网格状态 (9×4=36D, I=mean(RGB))
- Delta: 状态变化（下一时刻 - 当前时刻）
- 通过后验加权混合 (BMA) / MAP 选择 / 随机采样预测下一状态

推理模式:
    mode="mean"   : 后验加权混合 Σ w_k μ_k（平滑，适合多步模拟）
    mode="map"    : 选 top-1 规则 μ_{k*}（确定性，避免模糊）
    mode="sample" : 按权重采样规则，可选采样 Δ ~ N(μ_k, Σ_k)

Example:
    engine = PhysicsEngine()
    
    # 学习重力规则：红+绿 → 红下落
    engine.learn(grid_t, grid_next, "gravity")
    
    # 学习悬浮规则：红+蓝 → 红上升
    engine.learn(grid_t, grid_next, "levitate")
    
    # 预测：根据当前场景自动选择规则
    next_grid = engine.predict(current_grid, mode="mean")
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from concepts import ConceptTable, Concept
from scoring import log_inc_single
from templates import COLORS_RGB


# =============================================================================
# Physics Grid (3x3 RGBI)
# =============================================================================

class PhysicsGrid:
    """
    3x3 RGBI 物理网格。
    
    每个格子可以有一个颜色（或为空）。
    内部表示为 (3, 3, 4) 的 numpy 数组，最后一维是 RGBI (0-1)。
    I = mean(R, G, B)，是颜色无关的亮度/结构信息。
    """
    
    def __init__(self, data: Optional[np.ndarray] = None):
        """
        初始化网格。
        
        Args:
            data: (3, 3, 4) RGBI numpy 数组，或 (3, 3, 3) RGB 数组（自动补 I），
                  或 None 表示空网格
        """
        if data is None:
            self.data = np.zeros((3, 3, 4), dtype=np.float64)
        else:
            data = np.asarray(data, dtype=np.float64)
            if data.shape == (3, 3, 3):
                # 自动补 Intensity 通道
                intensity = data.mean(axis=2, keepdims=True)
                data = np.concatenate([data, intensity], axis=2)
            assert data.shape == (3, 3, 4), f"Grid must be (3, 3, 4), got {data.shape}"
            self.data = data
    
    @classmethod
    def from_colors(cls, positions: Dict[Tuple[int, int], str]) -> 'PhysicsGrid':
        """
        从颜色名称字典创建网格。
        
        Args:
            positions: {(row, col): color_name} 字典
                      row, col 范围是 0-2
                      
        Example:
            grid = PhysicsGrid.from_colors({
                (0, 1): 'red',    # 顶部中间放红色
                (2, 1): 'green',  # 底部中间放绿色
            })
        """
        grid = cls()
        for (row, col), color in positions.items():
            if color in COLORS_RGB:
                rgb = np.array(COLORS_RGB[color], dtype=np.float64) / 255.0
                grid.data[row, col, :3] = rgb
                grid.data[row, col, 3] = np.mean(rgb)  # I = mean(R,G,B)
        return grid
    
    def to_array(self) -> np.ndarray:
        """返回 (3, 3, 4) RGBI numpy 数组。"""
        return self.data.copy()
    
    def copy(self) -> 'PhysicsGrid':
        """深拷贝。"""
        return PhysicsGrid(self.data.copy())
    
    def get(self, row: int, col: int) -> np.ndarray:
        """获取某格子的 RGBI 值 (4,)。"""
        return self.data[row, col, :]
    
    def set(self, row: int, col: int, rgb: np.ndarray):
        """设置某格子的颜色值（接受 RGB(3) 或 RGBI(4)）。"""
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.shape == (3,):
            self.data[row, col, :3] = rgb
            self.data[row, col, 3] = np.mean(rgb)
        else:
            self.data[row, col] = rgb
    
    def clear(self, row: int, col: int):
        """清空某格子。"""
        self.data[row, col, :] = 0.0
    
    def is_empty(self, row: int, col: int) -> bool:
        """判断格子是否为空。"""
        return np.sum(self.data[row, col, :]) < 0.01
    
    def visualize(self, ax=None, title: str = ""):
        """
        可视化网格（只显示 RGB 通道，I 通道不显示）。
        
        Args:
            ax: matplotlib axes，如果为 None 则创建新图
            title: 标题
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # 画网格线
        for i in range(4):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5)
        
        # 画每个格子 (只用 RGB 前3通道)
        for row in range(3):
            for col in range(3):
                rgb = self.data[row, col, :3]  # 只取 RGB
                if np.sum(rgb) > 0.01:  # 非空
                    rect = mpatches.Rectangle(
                        (col - 0.4, row - 0.4), 0.8, 0.8,
                        facecolor=rgb, edgecolor='black', linewidth=1
                    )
                    ax.add_patch(rect)
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_title(title)
        
        if ax is None:
            plt.show()
    
    def __repr__(self):
        non_empty = []
        for row in range(3):
            for col in range(3):
                if np.sum(self.data[row, col, :3]) > 0.05:
                    non_empty.append(f"({row},{col})")
        return f"PhysicsGrid({', '.join(non_empty) if non_empty else 'empty'})"


# =============================================================================
# Physics Encoder
# =============================================================================

class PhysicsEncoder:
    """
    物理状态编码器。
    
    将 3x3 RGBI 网格编码为向量：
    - State: 36D (3x3x4 展平，带高斯模糊，I=mean(RGB))
    - Delta: 36D (下一帧 - 当前帧)
    - Joint: 72D (State + Delta 拼接)
    """
    
    def __init__(self, sigma: float = 0.5):
        """
        Args:
            sigma: 高斯模糊的标准差，用于空间容错
        """
        self.sigma = sigma
    
    def encode_state(self, grid: PhysicsGrid) -> np.ndarray:
        """
        编码当前状态为 36D 向量。
        
        对每个 RGBI 通道分别做高斯模糊，模拟"感受野"。
        I 通道在模糊后重新计算为 mean(blurred RGB)。
        
        Args:
            grid: PhysicsGrid 对象
            
        Returns:
            36D numpy 数组
        """
        data = grid.to_array()  # (3, 3, 4)
        blurred = np.zeros_like(data)
        
        # 模糊 RGB 通道
        for c in range(3):
            blurred[:, :, c] = gaussian_filter(data[:, :, c], sigma=self.sigma)
        
        # I 通道 = 模糊后 RGB 的均值
        blurred[:, :, 3] = blurred[:, :, :3].mean(axis=2)
        
        return blurred.flatten()
    
    def encode_delta(self, grid_t: PhysicsGrid, grid_next: PhysicsGrid) -> np.ndarray:
        """
        编码状态变化为 36D 向量。
        
        Args:
            grid_t: 当前时刻网格
            grid_next: 下一时刻网格
            
        Returns:
            36D numpy 数组
        """
        delta = grid_next.to_array() - grid_t.to_array()
        return delta.flatten()
    
    def encode_step(self, grid_t: PhysicsGrid, grid_next: PhysicsGrid) -> np.ndarray:
        """
        编码完整的物理步骤为 72D 向量。
        
        前36维: 当前状态 (Context, 什么条件下发生)
        后36维: 状态变化 (Effect, 发生了什么)
        
        Args:
            grid_t: 当前时刻网格
            grid_next: 下一时刻网格
            
        Returns:
            72D numpy 数组 [State, Delta]
        """
        state = self.encode_state(grid_t)
        delta = self.encode_delta(grid_t, grid_next)
        return np.concatenate([state, delta])
    
    def decode_delta(self, delta_vec: np.ndarray) -> np.ndarray:
        """
        将 36D delta 向量解码为 (3, 3, 4) 数组。
        
        Args:
            delta_vec: 36D numpy 数组
            
        Returns:
            (3, 3, 4) numpy 数组
        """
        return delta_vec.reshape(3, 3, 4)


# =============================================================================
# Physics Engine
# =============================================================================

class PhysicsEngine:
    """
    RSA 物理引擎 - 学习和预测条件物理规则。
    
    核心思想：
    - 物理规则是 54D 高斯概念 [Context, Effect]
    - Context: 什么条件下规则生效 (当前状态)
    - Effect: 规则的效果 (状态变化)
    - 通过后验加权混合 (BMA) / MAP / 采样预测下一状态
    
    推理模式：
    - mode="mean"  : 后验加权混合 Σ w_k μ_k (BMA, 默认)
    - mode="map"   : 选 top-1 规则 μ_{k*}
    - mode="sample": 按权重采样规则 + 采样 Δ ~ N(μ_k, Σ_k)
    
    可选 rho 正则: 偏好低方差 (更确信) 的规则
    
    Example:
        engine = PhysicsEngine()
        
        # 学习重力
        engine.learn(grid_t, grid_drop, "gravity")
        
        # 学习悬浮
        engine.learn(grid_t, grid_rise, "levitate")
        
        # 预测 (三种模式)
        next_grid, info = engine.predict(current_grid, mode="mean")
        next_grid, info = engine.predict(current_grid, mode="map")
        next_grid, info = engine.predict(current_grid, mode="sample")
    """
    
    def __init__(self, sigma: float = 0.5, temp: float = 0.1, rho: float = 0.0):
        """
        Args:
            sigma: 状态编码的高斯模糊标准差
            temp: Softmax 温度，控制规则选择的锐度
                  0.01 = 只选最匹配的规则
                  0.1  = 软加权 (默认)
                  1.0  = 均匀混合
            rho: Effect 不确定性正则系数 (默认 0.0 = 关闭)
                 当 rho > 0 时，低方差（更确信）的规则获得额外加分：
                 score += -rho * log|Σ_eff|
                 这是一个先验偏好/正则项，不是 precision-weighted mixture。
        """
        self.encoder = PhysicsEncoder(sigma=sigma)
        self.table = ConceptTable(d=72)  # 72维概念表 (36 Context + 36 Effect)
        self.temp = temp
        self.rho = rho
        self._train_count = 0
        
        print(f"Physics Engine 初始化完成 (σ={sigma}, temp={temp}, ρ={rho})")
    
    def learn(self, grid_t: PhysicsGrid, grid_next: PhysicsGrid, 
              rule_name: str, verbose: bool = True):
        """
        学习一个物理规则。
        
        Args:
            grid_t: 当前时刻网格
            grid_next: 下一时刻网格
            rule_name: 规则名称 (如 "gravity", "levitate")
            verbose: 是否打印学习信息
        """
        # 编码物理步骤
        vec = self.encoder.encode_step(grid_t, grid_next)
        
        # 直接更新概念（类似 Welford 在线更新）
        concept = self.table.ensure(rule_name)
        
        # 在线更新
        concept.kappa += 1.0
        delta = vec - concept.mu
        concept.mu = concept.mu + delta / concept.kappa
        
        # 更新方差
        if concept.kappa > 1:
            concept.var = concept.var + (delta * (vec - concept.mu) - concept.var) / concept.kappa
            concept.var = np.maximum(concept.var, 1e-6)  # 下限
        
        self._train_count += 1
        
        if verbose:
            print(f"[Learn] '{rule_name}' kappa={concept.kappa:.0f}")
    
    def predict(self, grid_t: PhysicsGrid,
                mode: str = "mean",
                verbose: bool = True) -> Tuple[PhysicsGrid, Dict]:
        """
        预测下一时刻的网格状态。
        
        流程：
        1. 编码当前状态
        2. 计算与每个规则的 Context 匹配度（+ 可选 rho 正则）
        3. 根据 mode 生成预测的 Effect
        4. 应用变化
        
        Args:
            grid_t: 当前时刻网格
            mode: 推理模式
                  "mean"   — 后验加权混合 Σ w_k μ_k (BMA, 默认)
                  "map"    — 选 top-1 规则 μ_{k*}
                  "sample" — 按权重采样规则，再采样 Δ ~ N(μ_k, Σ_k)
            verbose: 是否打印预测信息
            
        Returns:
            (predicted_grid, info_dict)
            info_dict 包含 scores, weights, delta, mode, chosen_rule (map/sample)
        """
        if mode not in ("mean", "map", "sample"):
            raise ValueError(f"mode 必须是 'mean', 'map', 'sample' 之一，收到: '{mode}'")
        
        if len(self.table._concepts) == 0:
            print("[Predict] 没有学习任何规则！")
            return grid_t.copy(), {}
        
        # 1. 编码当前状态
        state = self.encoder.encode_state(grid_t)
        
        # 2. 计算每个规则的匹配分数 (只比较前36维 Context)
        scores = {}
        for name, concept in self.table._concepts.items():
            # 只用 Context 部分 (前36维 RGBI) 计算似然
            ctx_mu = concept.mu[:36]
            ctx_var = concept.var[:36]
            score = log_inc_single(state, ctx_mu, ctx_var)
            
            # 可选：Effect 不确定性正则（先验偏好，不是 precision-weighted mixture）
            if self.rho > 0:
                eff_var = concept.var[36:]
                score -= self.rho * np.sum(np.log(eff_var + 1e-9))
            
            scores[name] = score
        
        # 3. Softmax 得到权重
        names = list(scores.keys())
        score_array = np.array([scores[n] for n in names])
        weight_array = softmax(score_array / self.temp)
        weights = {n: w for n, w in zip(names, weight_array)}
        
        # 4. 根据 mode 生成 Effect
        chosen_rule = None
        
        if mode == "mean":
            # BMA: 后验加权混合 Σ w_k μ_k
            pred_delta = np.zeros(36)
            for name, w in weights.items():
                effect_mu = self.table._concepts[name].mu[36:]
                pred_delta += w * effect_mu
        
        elif mode == "map":
            # MAP: 选 top-1 规则
            chosen_rule = max(weights, key=weights.get)
            pred_delta = self.table._concepts[chosen_rule].mu[36:].copy()
        
        elif mode == "sample":
            # Sample: 按权重采样规则，再从 N(μ_k, diag(σ²_k)) 采样
            chosen_idx = np.random.choice(len(names), p=weight_array)
            chosen_rule = names[chosen_idx]
            eff_mu = self.table._concepts[chosen_rule].mu[36:]
            eff_std = np.sqrt(self.table._concepts[chosen_rule].var[36:])
            pred_delta = eff_mu + eff_std * np.random.randn(36)
        
        # 5. 应用变化
        current_data = grid_t.to_array()
        delta_grid = self.encoder.decode_delta(pred_delta)
        next_data = current_data + delta_grid
        
        # 裁剪到 [0, 1]
        next_data = np.clip(next_data, 0.0, 1.0)
        
        # 过滤微小变化
        next_data[np.abs(next_data) < 0.05] = 0.0
        
        if verbose:
            top_rule = max(weights, key=weights.get)
            mode_label = {"mean": "BMA", "map": "MAP", "sample": "Sample"}[mode]
            print(f"[Predict/{mode_label}] Top rule: '{top_rule}' ({weights[top_rule]:.1%})")
            if chosen_rule and chosen_rule != top_rule:
                print(f"          Chosen (sampled): '{chosen_rule}' ({weights[chosen_rule]:.1%})")
            for name, w in sorted(weights.items(), key=lambda x: -x[1]):
                print(f"          {name}: {w:.1%} (score={scores[name]:.1f})")
        
        info = {
            "scores": scores,
            "weights": weights,
            "delta": pred_delta,
            "mode": mode,
        }
        if chosen_rule is not None:
            info["chosen_rule"] = chosen_rule
        
        return PhysicsGrid(next_data), info
    
    def simulate(self, grid_t: PhysicsGrid, steps: int = 5,
                 verbose: bool = True) -> List[PhysicsGrid]:
        """
        多步模拟。
        
        Args:
            grid_t: 初始网格
            steps: 模拟步数
            verbose: 是否打印每步信息
            
        Returns:
            网格序列 [t, t+1, t+2, ...]
        """
        trajectory = [grid_t.copy()]
        current = grid_t
        
        for i in range(steps):
            if verbose:
                print(f"\n--- Step {i+1}/{steps} ---")
            next_grid, info = self.predict(current, verbose=verbose)
            trajectory.append(next_grid)
            current = next_grid
        
        return trajectory
    
    def visualize_trajectory(self, trajectory: List[PhysicsGrid], 
                              cols: int = 5):
        """
        可视化轨迹序列。
        
        Args:
            trajectory: PhysicsGrid 列表
            cols: 每行显示的列数
        """
        n = len(trajectory)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_2d(axes)
        
        for i, grid in enumerate(trajectory):
            row, col = divmod(i, cols)
            grid.visualize(ax=axes[row, col], title=f"t={i}")
        
        # 隐藏多余的子图
        for i in range(n, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def laws(self) -> List[str]:
        """返回所有已学习的规则名称。"""
        return list(self.table._concepts.keys())
    
    def status(self):
        """打印引擎状态。"""
        print(f"\n{'='*40}")
        print("Physics Engine 状态")
        print(f"{'='*40}")
        print(f"训练样本数: {self._train_count}")
        print(f"已学规则: {self.laws()}")
        print(f"编码维度: 72 (State=36 + Delta=36, RGBI)")
        print(f"σ (空间模糊): {self.encoder.sigma}")
        print(f"temp (选择锐度): {self.temp}")
        print(f"ρ (Effect不确定性正则): {self.rho}")
        print(f"推理模式: mean (BMA) / map (top-1) / sample (随机)")
        print()
    
    def reset(self):
        """重置引擎。"""
        self.table = ConceptTable(d=72)
        self._train_count = 0
        print("[Reset] Physics Engine 已重置")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """
    演示：条件物理规则学习。
    
    场景：
    - 红+绿 → 红下落 (gravity)
    - 红+蓝 → 红上升 (levitate)
    """
    print("="*60)
    print(" RSA Physics Engine Demo")
    print("="*60)
    
    engine = PhysicsEngine(sigma=0.5, temp=0.1)
    
    # ========== 学习重力规则 ==========
    print("\n--- 学习 Gravity 规则: 红+绿 → 红下落 ---")
    
    # 红在上，绿在下
    grid_t1 = PhysicsGrid.from_colors({(0, 1): 'red', (2, 1): 'green'})
    # 红下落一格
    grid_t2 = PhysicsGrid.from_colors({(1, 1): 'red', (2, 1): 'green'})
    
    engine.learn(grid_t1, grid_t2, "gravity")
    
    # 多训练几次
    grid_t3 = PhysicsGrid.from_colors({(1, 1): 'red', (2, 1): 'green'})
    grid_t4 = PhysicsGrid.from_colors({(2, 1): 'red', (2, 1): 'green'})  # 红到底
    engine.learn(grid_t3, grid_t4, "gravity")
    
    # ========== 学习悬浮规则 ==========
    print("\n--- 学习 Levitate 规则: 红+蓝 → 红上升 ---")
    
    # 红在下，蓝在右
    grid_l1 = PhysicsGrid.from_colors({(2, 1): 'red', (1, 2): 'blue'})
    # 红上升一格
    grid_l2 = PhysicsGrid.from_colors({(1, 1): 'red', (1, 2): 'blue'})
    
    engine.learn(grid_l1, grid_l2, "levitate")
    
    grid_l3 = PhysicsGrid.from_colors({(1, 1): 'red', (1, 2): 'blue'})
    grid_l4 = PhysicsGrid.from_colors({(0, 1): 'red', (1, 2): 'blue'})
    engine.learn(grid_l3, grid_l4, "levitate")
    
    # ========== 条件推理测试 ==========
    print("\n--- 测试条件推理 ---")
    
    # 场景 A: 有绿色 → 应该触发 gravity
    print("\n场景 A: 红+绿 (应触发 gravity)")
    test_a = PhysicsGrid.from_colors({(0, 1): 'red', (2, 0): 'green'})
    pred_a, info_a = engine.predict(test_a)
    
    # 场景 B: 有蓝色 → 应该触发 levitate
    print("\n场景 B: 红+蓝 (应触发 levitate)")
    test_b = PhysicsGrid.from_colors({(2, 1): 'red', (1, 2): 'blue'})
    pred_b, info_b = engine.predict(test_b)
    
    engine.status()
    
    print("\nDemo 完成!")


if __name__ == "__main__":
    demo()
