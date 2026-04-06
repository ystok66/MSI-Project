"""
RSA Pixel Motor System - 36D RGBI 像素空间动作系统

核心思想：
- 使用 3x3 网格 × 4通道(R, G, B, Intensity) 的 36 维向量
- 第 4 通道 Intensity = mean(R, G, B)，捕获颜色无关的结构信息
- 颜色概念学习"什么颜色"（位置无关）
- 位置概念学习"哪个位置"（颜色无关）
- 零样本组合：通过高斯专家乘积实现约束交集

关键机制：
- Product of Gaussian Experts: μ_new = (Σ μ_i * prec_i) / Σ prec_i
- 方差作为"否决权"：方差小 = 确信度高 = 话语权大
- 信号增强：归一化幸存信号

Example:
    agent = PixelMotorSystem()
    agent.babble()  # 牙牙学语训练
    grid = agent.imagine(['cmd_red', 'cmd_pos_4'])  # 零样本组合
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from concepts import ConceptTable


class PixelMotorSystem:
    """
    36维 RGBI 像素空间的运动控制系统。
    
    每个 cell = [R, G, B, I]，I = mean(R,G,B)
    9 cells × 4 channels = 36D
    
    实现：
    - 牙牙学语 (Motor Babbling): 随机探索建立概念
    - 意图合成 (Imagine): 高斯专家乘积实现零样本组合
    """
    
    def __init__(self, color_samples: int = 500, pos_samples: int = 100):
        """
        Args:
            color_samples: 每个颜色概念的训练样本数
            pos_samples: 每个位置概念的训练样本数
        """
        # 36维动作概念表: 9个格子 * 4通道(R,G,B,I)
        self.motor_cortex = ConceptTable(d=36)
        
        self.color_samples = color_samples
        self.pos_samples = pos_samples
        
        # 基础指令集
        self.color_cmds = ['cmd_red', 'cmd_yellow', 'cmd_blue', 'cmd_green', 
                          'cmd_cyan', 'cmd_magenta', 'cmd_orange', 'cmd_purple']
        self.pos_cmds = [f'cmd_pos_{i}' for i in range(9)]
        
        # 颜色定义 (RGB, 0-1)
        self.palette = {
            'cmd_red':     [1.0, 0.0, 0.0],
            'cmd_yellow':  [1.0, 1.0, 0.0],
            'cmd_blue':    [0.0, 0.0, 1.0],
            'cmd_green':   [0.0, 1.0, 0.0],
            'cmd_cyan':    [0.0, 1.0, 1.0],
            'cmd_magenta': [1.0, 0.0, 1.0],
            'cmd_orange':  [1.0, 0.5, 0.0],
            'cmd_purple':  [0.5, 0.0, 1.0],
        }
        
        self._trained = False
        print(f"PixelMotorSystem 初始化完成 (d=36)")
    
    def _create_grid_vec(self, rgb: List[float], pos_idx: int) -> np.ndarray:
        """
        生成单次动作的视觉结果 (36维向量)。
        
        Args:
            rgb: [R, G, B] 颜色值
            pos_idx: 位置索引 (0-8)
            
        Returns:
            36维向量 (9×4: RGBI)，只有 pos_idx 位置有颜色
        """
        grid = np.zeros((9, 4))
        grid[pos_idx, :3] = rgb
        grid[pos_idx, 3] = np.mean(rgb)  # intensity = mean(R,G,B)
        return grid.flatten()
    
    def _update_concept(self, vec: np.ndarray, token: str):
        """
        直接在线更新概念（Welford算法）。
        
        Args:
            vec: 36维特征向量
            token: 概念名称
        """
        concept = self.motor_cortex.ensure(token)
        
        concept.kappa += 1.0
        delta = vec - concept.mu
        concept.mu = concept.mu + delta / concept.kappa
        
        if concept.kappa > 1:
            concept.var = concept.var + (delta * (vec - concept.mu) - concept.var) / concept.kappa
            concept.var = np.maximum(concept.var, 1e-8)
    
    def babble(self, verbose: bool = True):
        """
        👶 运动牙牙学语 (Motor Babbling)
        
        阶段 1: 学习颜色概念
        - 固定颜色，随机位置
        - 学到: "红色" = GB通道恒为0（低方差），R通道位置不定（高方差）
        
        阶段 2: 学习位置概念
        - 固定位置，随机颜色
        - 学到: "位置4" = 只有中心有值（其他位置低方差为0）
        """
        if verbose:
            print("🚀 [36D RGBI Cortex] Starting Motor Babbling...")
        
        # === 阶段 1: 学习颜色概念 ===
        if verbose:
            print(f"   Phase 1: Learning {len(self.palette)} colors ({self.color_samples} samples each)")
        
        for c_cmd in self.palette.keys():
            target_rgb = self.palette[c_cmd]
            for _ in range(self.color_samples):
                rand_pos = np.random.randint(0, 9)
                vec = self._create_grid_vec(target_rgb, rand_pos)
                self._update_concept(vec, c_cmd)
        
        # === 阶段 2: 学习位置概念 ===
        if verbose:
            print(f"   Phase 2: Learning 9 positions ({self.pos_samples} samples each)")
        
        for p_idx in range(9):
            p_cmd = f'cmd_pos_{p_idx}'
            for _ in range(self.pos_samples):
                rand_color_key = np.random.choice(list(self.palette.keys()))
                rand_rgb = self.palette[rand_color_key]
                vec = self._create_grid_vec(rand_rgb, p_idx)
                self._update_concept(vec, p_cmd)
        
        self._trained = True
        if verbose:
            print(f"✅ Cortex Calibrated. Learned {len(self.motor_cortex._concepts)} motor concepts.")
    
    def imagine(self, commands: List[str], 
                visualize: bool = False,
                amplify: bool = True) -> np.ndarray:
        """
        🧠 意图合成: Product of Gaussian Experts
        
        通过精度加权平均实现零样本组合。
        方差小的通道有更大话语权，形成"约束的交集"。
        
        Args:
            commands: 命令列表，如 ['cmd_red', 'cmd_pos_4']
            visualize: 是否显示结果
            amplify: 是否进行信号增强
            
        Returns:
            (3, 3, 4) RGBI 网格
        """
        if not self._trained:
            print("⚠️ 请先调用 babble() 进行训练！")
            return np.zeros((3, 3, 4))
        
        # 过滤有效指令
        valid_tokens = [t for t in commands if t in self.motor_cortex._concepts]
        if not valid_tokens:
            print(f"⚠️ 没有有效指令: {commands}")
            return np.zeros((3, 3, 4))
        
        means = []
        precs = []  # Precision = 1 / Variance
        
        for token in valid_tokens:
            c = self.motor_cortex._concepts[token]
            means.append(c.mu)
            # 方差越小 → 精度越高 → 话语权越大
            precs.append(1.0 / (c.var + 1e-9))
        
        means = np.array(means)
        precs = np.array(precs)
        
        # === 核心公式: 精度加权平均 ===
        # μ_new = (Σ μ_i * prec_i) / Σ prec_i
        total_prec = np.sum(precs, axis=0)
        weighted_mu = np.sum(means * precs, axis=0)
        joint_mu = weighted_mu / total_prec
        
        # === 信号增强 ===
        # 因为颜色被位置稀释了，需要归一化
        final_vec = joint_mu.copy()
        if amplify and final_vec.max() > 0.01:
            final_vec = final_vec / final_vec.max()
        
        # 裁剪到 [0, 1]
        final_vec = np.clip(final_vec, 0.0, 1.0)
        
        # 重塑为 3x3x4 (RGBI)
        grid = final_vec.reshape(3, 3, 4)
        
        if visualize:
            self.visualize_grid(grid, title=f"Imagine: {valid_tokens}")
        
        return grid
    
    def inspect(self, token: str, amplify_factor: float = 5.0):
        """
        🔍 检查某个概念的内部表示。
        
        Args:
            token: 概念名称
            amplify_factor: 亮度放大系数
        """
        if token not in self.motor_cortex._concepts:
            print(f"⚠️ 概念 '{token}' 不存在")
            return
        
        concept = self.motor_cortex._concepts[token]
        grid = concept.mu.reshape(3, 3, 4)
        
        print(f"\n🔍 Concept '{token}':")
        print(f"   κ = {concept.kappa:.0f}")
        print(f"   μ range: [{concept.mu.min():.3f}, {concept.mu.max():.3f}]")
        print(f"   σ² range: [{concept.var.min():.6f}, {concept.var.max():.3f}]")
        print(f"   Intensity (ch3) range: [{grid[:,:,3].min():.3f}, {grid[:,:,3].max():.3f}]")
        
        self.visualize_grid(grid * amplify_factor, 
                           title=f"Memory: {token} (×{amplify_factor})")
    
    def visualize_grid(self, grid: np.ndarray, title: str = ""):
        """
        可视化 3x3 RGBI 网格（显示 RGB 通道）。
        
        Args:
            grid: (3, 3, 4) 或 (3, 3, 3) numpy 数组
            title: 标题
        """
        # 提取 RGB 通道用于显示
        if grid.shape[-1] == 4:
            rgb = grid[:, :, :3]
        else:
            rgb = grid
        
        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(rgb, 0, 1))
        plt.title(title)
        plt.axis('off')
        
        # 添加网格线
        for i in range(4):
            plt.axhline(i - 0.5, color='gray', linewidth=0.5)
            plt.axvline(i - 0.5, color='gray', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def concepts(self) -> List[str]:
        """返回所有已学概念。"""
        return list(self.motor_cortex._concepts.keys())
    
    def status(self):
        """打印状态。"""
        print(f"\n{'='*40}")
        print("PixelMotorSystem 状态 (36D RGBI)")
        print(f"{'='*40}")
        print(f"已训练: {self._trained}")
        print(f"颜色概念: {[c for c in self.concepts() if c.startswith('cmd_') and not c.startswith('cmd_pos')]}")
        print(f"位置概念: {[c for c in self.concepts() if c.startswith('cmd_pos')]}")
        print(f"总概念数: {len(self.concepts())}")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """
    演示：零样本运动组合。
    
    Agent 从未见过 "Red at Pos 4" 的组合样本，
    但可以通过高斯专家乘积自动合成。
    """
    print("="*60)
    print(" PixelMotorSystem Demo: Zero-Shot Composition")
    print("="*60)
    
    # 1. 初始化
    agent = PixelMotorSystem(color_samples=500, pos_samples=100)
    
    # 2. 牙牙学语
    agent.babble()
    
    # 3. 检查内部表示
    print("\n--- 检查颜色概念 'cmd_red' ---")
    agent.inspect('cmd_red', amplify_factor=5.0)
    
    print("\n--- 检查位置概念 'cmd_pos_4' (中心) ---")
    agent.inspect('cmd_pos_4', amplify_factor=3.0)
    
    # 4. 零样本组合
    print("\n--- 零样本组合: Red + Pos 4 ---")
    print("Agent 从未见过这个组合，但可以自动合成！")
    grid1 = agent.imagine(['cmd_red', 'cmd_pos_4'], visualize=True)
    
    # 5. 多点组合
    print("\n--- 多点组合: Blue + Pos 0 + Pos 8 (对角线) ---")
    grid2 = agent.imagine(['cmd_blue', 'cmd_pos_0', 'cmd_pos_8'], visualize=True)
    
    # 6. 多颜色（混合）
    print("\n--- 颜色混合: Red + Blue + Pos 4 ---")
    grid3 = agent.imagine(['cmd_red', 'cmd_blue', 'cmd_pos_4'], visualize=True)
    
    agent.status()
    print("\nDemo 完成!")


if __name__ == "__main__":
    demo()
