"""
RSA Planning Agent - 基于模型的具身规划系统

核心能力：
1. 动态概念学习 - 学习状态变化模式（如 "grow"）
2. 层次贝叶斯 (Hierarchical Bayes) - 共享结构原型 + 颜色残差
   μ_eff,c = μ_shared + δ_c,  δ_c ~ N(0, σ²_δ I)
3. 逆向规划 - 想象-评分-贪婪搜索

设计思想：
    把每个颜色当作一个 task，用共享先验实现 few-shot。
    μ_shared 捕获跨颜色不变的空间结构（"哪里变亮"）；
    δ_c 捕获颜色特定的残差（通常很小）。
    新颜色 zero-shot 时 δ_c = 0（回归到 shared prior）；
    给 1-3 个样本就能将 δ_c 拉到位。

Example:
    agent = PlanningAgent()
    agent.babble()
    agent.learn_dynamic_concept(start, end, "grow")
    history = agent.ask_to_show(blue_start, "grow")
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from rsa_action_pixel import PixelMotorSystem
from concepts import ConceptTable, Concept
from scoring import log_inc_single


@dataclass
class ColorResidual:
    """颜色特定的残差 δ_c，带在线 Welford 更新。"""
    mu: np.ndarray          # 残差均值 (36D)
    var: np.ndarray         # 残差方差 (36D)
    kappa: float = 0.0      # 观测数


class PlanningAgent(PixelMotorSystem):
    """
    具身规划代理：层级贝叶斯学习 + 逆向规划。
    
    核心机制：
    - 层级贝叶斯 (Hierarchical Bayes):
      μ_eff,c = μ_shared + δ_c
      μ_shared: 跨颜色共享的结构变化原型（"哪里变亮"）
      δ_c:      颜色特定的残差，先验 N(0, σ²_δ I)
    - 语境感知 (Context Awareness): 自动检测物体颜色
    - 贝叶斯规划 (Bayesian Planning): 想象 → 评分 → 贪婪搜索
    """
    
    def __init__(self, sigma_delta: float = 0.01):
        """
        Args:
            sigma_delta: 残差先验方差 σ²_δ。越小 = shrinkage 越强。
                         0.01 = 强正则 (残差难以偏离 0, 默认)
                         0.1  = 中等正则
                         1.0  = 弱正则 (接近独立学习)
        """
        super().__init__()
        # 动态概念皮层 (36维 RGBI): 存储"变化模式"
        self.dynamic_concepts = ConceptTable(d=36)
        self._dynamic_count = 0
        
        # 层级贝叶斯参数
        self.sigma_delta = sigma_delta
        self._shared_concepts = ConceptTable(d=36)   # μ_shared
        self._residuals: Dict[str, Dict[str, ColorResidual]] = {}  # δ_c
        
        print(f"PlanningAgent 初始化完成 (σ²_δ={sigma_delta})")
    
    @staticmethod
    def _welford_update(mu: np.ndarray, var: np.ndarray,
                        kappa: float, x: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Welford 在线更新，返回 (new_mu, new_var, new_kappa)。"""
        kappa_new = kappa + 1.0
        delta = x - mu
        mu_new = mu + delta / kappa_new
        if kappa_new > 1:
            var_new = var + (delta * (x - mu_new) - var) / kappa_new
            var_new = np.maximum(var_new, 1e-8)
        else:
            var_new = var
        return mu_new, var_new, kappa_new
    
    def _update_dynamic_concept(self, vec: np.ndarray, token: str):
        """直接在线更新动态概念（兼容旧接口）。"""
        concept = self.dynamic_concepts.ensure(token)
        concept.mu, concept.var, concept.kappa = self._welford_update(
            concept.mu, concept.var, concept.kappa, vec)
    
    def _detect_color_key(self, grid: np.ndarray) -> str:
        """从网格检测主色调，返回 palette 中最近的 color key。"""
        # 提取 RGB 通道 (忽略第 4 通道 Intensity)
        rgb_grid = grid[:, :, :3] if grid.shape[-1] == 4 else grid
        mask = np.sum(rgb_grid, axis=2) > 0.1
        if not mask.any():
            return 'cmd_blue'
        avg_color = rgb_grid[mask].mean(axis=0)
        min_dist, best_key = 999, 'cmd_blue'
        for c_name, rgb in self.palette.items():
            d = np.linalg.norm(np.array(rgb) - avg_color)
            if d < min_dist:
                min_dist, best_key = d, c_name
        return best_key
    
    def _get_effective_concept(self, rule_name: str,
                                color_key: Optional[str] = None
                                ) -> Optional[Concept]:
        """
        构造有效概念: μ_eff = μ_shared + δ_c
        未见过的颜色 → δ_c = 0 (先验, zero-shot)
        """
        if rule_name not in self._shared_concepts._concepts:
            return None
        shared = self._shared_concepts._concepts[rule_name]
        
        if (color_key and rule_name in self._residuals
                and color_key in self._residuals[rule_name]):
            res = self._residuals[rule_name][color_key]
            eff_mu = shared.mu + res.mu
            eff_var = shared.var + res.var
        else:
            eff_mu = shared.mu.copy()
            eff_var = shared.var + self.sigma_delta
        
        return Concept(token=rule_name, mu=eff_mu, var=eff_var,
                       kappa=shared.kappa)
    
    def learn_dynamic_concept(self, start_grid: np.ndarray, end_grid: np.ndarray, 
                               name: str, verbose: bool = True):
        """
        🎓 层级贝叶斯学习动态概念 (如 'grow')
        
        μ_eff,c = μ_shared + δ_c
        - raw delta → 更新 μ_shared (跨颜色共享结构)
        - residual = raw - μ_shared → 更新 δ_c (颜色残差, with shrinkage)
        
        不使用认知白化，颜色信息被保留在残差中。
        
        Args:
            start_grid: (3, 3, 4) 初始状态
            end_grid: (3, 3, 4) 结束状态
            name: 概念名称，如 "grow"
            verbose: 是否打印信息
        """
        # 1. 计算 raw delta (无白化)
        raw_delta = np.maximum(end_grid - start_grid, 0)
        raw_vec = raw_delta.flatten()  # 36D
        
        # 2. 更新 μ_shared (跨颜色共享)
        shared = self._shared_concepts.ensure(name)
        shared.mu, shared.var, shared.kappa = self._welford_update(
            shared.mu, shared.var, shared.kappa, raw_vec)
        
        # 3. 更新 δ_c (颜色残差)
        color_key = self._detect_color_key(end_grid)
        if name not in self._residuals:
            self._residuals[name] = {}
        if color_key not in self._residuals[name]:
            self._residuals[name][color_key] = ColorResidual(
                mu=np.zeros(36),
                var=np.full(36, self.sigma_delta),
                kappa=0.0,
            )
        
        res = self._residuals[name][color_key]
        residual_vec = raw_vec - shared.mu
        res.mu, res.var, res.kappa = self._welford_update(
            res.mu, res.var, res.kappa, residual_vec)
        res.var = np.maximum(res.var, self.sigma_delta)  # shrinkage 下限
        
        # 4. 同步更新 dynamic_concepts (兼容 compose_concepts 等)
        eff = self._get_effective_concept(name, color_key)
        if eff:
            dc = self.dynamic_concepts.ensure(name)
            dc.mu, dc.var, dc.kappa = eff.mu.copy(), eff.var.copy(), eff.kappa
        
        self._dynamic_count += 1
        
        if verbose:
            print(f"   📚 Learned '{name}' [{color_key}] "
                  f"(shared κ={shared.kappa:.0f}, "
                  f"δ_{color_key} κ={res.kappa:.0f})")
    
    def ask_to_show(self, start_grid: np.ndarray, goal_concept: str, 
                    max_steps: int = 10, verbose: bool = True) -> List[np.ndarray]:
        """
        🧠 Ask to Show (规划循环)
        
        层级贝叶斯版本：用 μ_eff = μ_shared + δ_c 作为目标概念。
        未见过的颜色 → δ_c = 0 (zero-shot, 回退到 shared prior)
        
        Args:
            start_grid: (3, 3, 4) 初始网格
            goal_concept: 目标概念名称，如 "grow"
            max_steps: 最大步数
            verbose: 是否打印详细信息
            
        Returns:
            网格历史列表
        """
        if verbose:
            print(f"\n🤔 Planning to show '{goal_concept}'...")
        
        # === 1. 语境感知 + 构造 effective concept ===
        color_key = self._detect_color_key(start_grid)
        target_concept = self._get_effective_concept(goal_concept, color_key)
        
        if target_concept is None:
            print(f"   ❌ Concept '{goal_concept}' unknown!")
            return [start_grid]
        
        has_residual = (goal_concept in self._residuals
                        and color_key in self._residuals.get(goal_concept, {}))
        
        if verbose:
            mode_str = 'few-shot' if has_residual else 'zero-shot ← shared prior'
            print(f"   🖌️ Context: {color_key} ({mode_str})")
        
        current_grid = start_grid.copy()
        accumulated_delta = np.zeros(36)
        history = [current_grid.copy()]
        
        # === 2. 规划循环 ===
        for step in range(max_steps):
            best_action = None
            best_score = -np.inf
            best_step_delta = None
            
            # A. 枚举候选动作 (只用当前颜色)
            for p_cmd in self.pos_cmds:
                action = [color_key, p_cmd]
                
                # 想象这个动作的效果
                step_delta_grid = self.imagine(action, visualize=False)
                step_delta_vec = step_delta_grid.flatten()
                
                # 假设执行这一步后的累积变化
                proposed_accumulated = accumulated_delta + step_delta_vec
                
                # 评分: 累积变化有多像 effective concept?
                score = log_inc_single(proposed_accumulated, 
                                       target_concept.mu, 
                                       target_concept.var)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_step_delta = step_delta_grid
            
            # C. 决策
            pos_idx = int(best_action[1].split('_')[-1])
            row, col = divmod(pos_idx, 3)
            is_occupied = np.sum(current_grid[row, col, :3]) > 0.1
            
            if is_occupied:
                if verbose:
                    print(f"   🛑 Converged at step {step}. ({best_action} hits occupied)")
                break
            
            if verbose:
                print(f"   Step {step+1}: {best_action} (score={best_score:.1f}) → Execute")
            
            # 执行动作
            current_grid = np.maximum(current_grid, best_step_delta)
            accumulated_delta += best_step_delta.flatten()
            history.append(current_grid.copy())
        
        if verbose:
            print(f"✅ Plan complete. Steps: {len(history)-1}")
        
        return history
    
    def visualize_plan(self, history: List[np.ndarray], title: str = "Plan"):
        """可视化规划过程。"""
        n = len(history)
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        if n == 1:
            axes = [axes]
        
        for i, grid in enumerate(history):
            # 显示 RGB 通道
            rgb = grid[:, :, :3] if grid.shape[-1] == 4 else grid
            axes[i].imshow(np.clip(rgb, 0, 1))
            axes[i].set_title(f"Step {i}")
            axes[i].axis('off')
            
            # 网格线
            for j in range(4):
                axes[i].axhline(j - 0.5, color='gray', linewidth=0.3)
                axes[i].axvline(j - 0.5, color='gray', linewidth=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def dynamic_concepts_list(self) -> List[str]:
        """返回已学动态概念列表。"""
        return list(self.dynamic_concepts._concepts.keys())
    
    def learn_visual_concept(self, target_grid: np.ndarray, name: str, 
                              whitening: bool = False, verbose: bool = True):
        """
        🎓 学习视觉概念 (从空白到目标状态)
        
        与 learn_dynamic_concept 不同，这个方法直接学习目标状态，
        而不是状态变化。用于学习 "blue"、"box" 等独立概念。
        
        Args:
            target_grid: (3, 3, 4) 目标状态
            name: 概念名称，如 "blue" 或 "box"
            whitening: 保留参数兼容性，不再使用
            verbose: 是否打印信息
        """
        vec = target_grid.flatten()
        
        self._update_dynamic_concept(vec, name)
        self._dynamic_count += 1
        
        if verbose:
            print(f"   📚 Learned '{name}' (κ={self.dynamic_concepts._concepts[name].kappa:.0f})")
    
    def compose_concepts(self, concept_names: List[str], new_name: str = None,
                          verbose: bool = True) -> Optional[np.ndarray]:
        """
        ⚗️ 概念组合：通过高斯乘积合成新概念
        
        核心数学：
        - 精度 (Precision) = 1 / 方差
        - 新均值 = Σ(μ_i * prec_i) / Σ(prec_i)
        - 方差小的概念拥有更大话语权 ("否决权")
        
        Args:
            concept_names: 要组合的概念名称列表，如 ["blue", "box"]
            new_name: 合成概念的名称，如果为 None 则自动生成
            verbose: 是否打印信息
            
        Returns:
            合成后的 (3, 3, 4) 网格，如果失败返回 None
        """
        # 验证所有概念存在
        valid_concepts = []
        for name in concept_names:
            if name in self.dynamic_concepts._concepts:
                valid_concepts.append(name)
            else:
                if verbose:
                    print(f"   ⚠️ Concept '{name}' not found, skipping")
        
        if len(valid_concepts) < 2:
            print("   ❌ Need at least 2 valid concepts to compose")
            return None
        
        if verbose:
            print(f"\n⚗️ Composing: {valid_concepts}")
        
        # 收集均值和精度
        eps = 1e-9
        means = []
        precs = []
        
        for name in valid_concepts:
            c = self.dynamic_concepts._concepts[name]
            means.append(c.mu)
            precs.append(1.0 / (c.var + eps))
        
        means = np.array(means)
        precs = np.array(precs)
        
        # 高斯乘积公式
        total_prec = np.sum(precs, axis=0)
        weighted_mu = np.sum(means * precs, axis=0)
        mu_new = weighted_mu / total_prec
        var_new = 1.0 / total_prec
        
        # 保存合成概念
        if new_name is None:
            new_name = "_".join(valid_concepts)
        
        # 创建新概念并注入
        from concepts import Concept
        synthetic = Concept(token=new_name, mu=mu_new, var=var_new, kappa=10.0)
        self.dynamic_concepts._concepts[new_name] = synthetic
        
        if verbose:
            print(f"   ✅ Created '{new_name}'")
            for name in valid_concepts:
                c = self.dynamic_concepts._concepts[name]
                var_mean = np.mean(c.var)
                print(f"      - {name}: avg_var={var_mean:.4f}")
        
        # 返回可视化用的网格
        result = mu_new.reshape(3, 3, 4)
        # 信号增强
        if result.max() > 0.01:
            result = result / result.max()
        result = np.clip(result, 0, 1)
        
        return result
    
    def generate(self, target_concept: str, color_cmd: str = None,
                 verbose: bool = True) -> List[np.ndarray]:
        """
        🎨 生成式规划：从空白画布生成目标概念
        
        使用层级贝叶斯的 effective concept 作为规划目标。
        
        Args:
            target_concept: 目标概念名称
            color_cmd: 使用的颜色命令，如 'cmd_blue'。如果为 None 则自动检测
            verbose: 是否打印信息
            
        Returns:
            生成过程的网格历史
        """
        blank = np.zeros((3, 3, 4))
        
        # 如果没有指定颜色，从 shared concept 检测
        if color_cmd is None:
            if target_concept in self._shared_concepts._concepts:
                shared = self._shared_concepts._concepts[target_concept]
                target_grid = shared.mu.reshape(3, 3, 4)
                avg_color = np.mean(target_grid[:, :, :3], axis=(0, 1))
                min_dist, color_cmd = 999, 'cmd_blue'
                for c_name, rgb in self.palette.items():
                    d = np.linalg.norm(np.array(rgb) - avg_color)
                    if d < min_dist:
                        min_dist, color_cmd = d, c_name
            else:
                color_cmd = 'cmd_blue'
        
        # 构造 effective concept (先查层级贝叶斯，再查 dynamic_concepts)
        target = self._get_effective_concept(target_concept, color_cmd)
        if target is None:
            # Fallback: 直接从 dynamic_concepts 取（compose_concepts 存在这里）
            if target_concept in self.dynamic_concepts._concepts:
                target = self.dynamic_concepts._concepts[target_concept]
            else:
                print(f"   ❌ Concept '{target_concept}' unknown!")
                return [blank]
        
        if verbose:
            print(f"\n🎨 Generating '{target_concept}' with {color_cmd}...")
        
        # 贪婪规划
        current = blank.copy()
        accumulated = np.zeros(36)
        history = [current.copy()]
        
        for step in range(9):
            best_action = None
            best_score = -np.inf
            best_delta = None
            
            for p_cmd in self.pos_cmds:
                action = [color_cmd, p_cmd]
                step_delta = self.imagine(action, visualize=False)
                proposed = accumulated + step_delta.flatten()
                
                score = log_inc_single(proposed, target.mu, target.var)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_delta = step_delta
            
            pos_idx = int(best_action[1].split('_')[-1])
            row, col = divmod(pos_idx, 3)
            if np.sum(current[row, col, :3]) > 0.1:
                if verbose:
                    print(f"   🛑 Converged at step {step}")
                break
            
            if verbose:
                print(f"   Step {step+1}: {best_action[1]} (score={best_score:.1f})")
            
            current = np.maximum(current, best_delta)
            accumulated += best_delta.flatten()
            history.append(current.copy())
        
        if verbose:
            print(f"✅ Generation complete. Steps: {len(history)-1}")
        
        return history


# =============================================================================
# Helper Functions
# =============================================================================

def make_scene(color: List[float], shape: str = 'dot') -> np.ndarray:
    """
    创建测试场景。
    
    Args:
        color: [R, G, B] 颜色值 (0-1)
        shape: 形状名称
               - 'dot': 中心点
               - 'cross': 十字形
               - 'box': 空心方框
               - 'solid': 实心方块
               - 'l': L形
               - 'corner': 角形
               - 'hbar': 水平条
               - 'vbar': 垂直条
    """
    g = np.zeros((3, 3, 4))
    color = np.array(color)
    intensity = np.mean(color)  # 第 4 通道
    
    # 形状定义 (填充的位置列表)
    shapes = {
        'dot':    [(1, 1)],
        'cross':  [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        'box':    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
        'solid':  [(r, c) for r in range(3) for c in range(3)],
        'l':      [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        'corner': [(0, 0), (0, 1), (1, 0)],
        'hbar':   [(1, 0), (1, 1), (1, 2)],
        'vbar':   [(0, 1), (1, 1), (2, 1)],
        't':      [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        'diag':   [(0, 0), (1, 1), (2, 2)],
    }
    
    positions = shapes.get(shape, [(1, 1)])  # 默认 dot
    for (r, c) in positions:
        g[r, c, :3] = color
        g[r, c, 3] = intensity
    
    return g


# =============================================================================
# Demo
# =============================================================================

def demo():
    """
    演示：层级贝叶斯 few-shot 跨颜色泛化。
    
    1. 用红/黄/绿教 "grow" (dot → cross)
    2. 给蓝色 3 个 few-shot 样本
    3. 测试蓝色 grow
    """
    print("="*60)
    print(" PlanningAgent Demo: Hierarchical Bayesian Few-Shot")
    print("="*60)
    
    agent = PlanningAgent(sigma_delta=0.01)
    
    # 1. 运动预热
    agent.babble()
    
    # 2. 教学阶段
    print("\n🎓 Teacher: Showing 'grow' with Red, Yellow, Green...")
    
    red = [1, 0, 0]
    yellow = [1, 1, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    
    # 训练 "grow" 概念 (多颜色)
    for _ in range(20):
        agent.learn_dynamic_concept(make_scene(red, 'dot'), make_scene(red, 'cross'), "grow", verbose=False)
        agent.learn_dynamic_concept(make_scene(yellow, 'dot'), make_scene(yellow, 'cross'), "grow", verbose=False)
        agent.learn_dynamic_concept(make_scene(green, 'dot'), make_scene(green, 'cross'), "grow", verbose=False)
    
    shared = agent._shared_concepts._concepts['grow']
    print(f"   ✅ 'grow' shared learned (κ={shared.kappa:.0f})")
    print(f"      Residuals: {list(agent._residuals.get('grow', {}).keys())}")
    
    # 3. Few-shot: 蓝色 3 个样本
    print("\n🔵 Few-shot: 3 blue samples...")
    for _ in range(3):
        agent.learn_dynamic_concept(make_scene(blue, 'dot'), make_scene(blue, 'cross'), "grow", verbose=False)
    print("   ✅ Blue residual δ_blue learned")
    
    # 4. 测试
    print("\n🧪 Test: 'grow' on BLUE point (few-shot)")
    blue_start = make_scene(blue, 'dot')
    history = agent.ask_to_show(blue_start, "grow")
    
    # 5. 可视化
    agent.visualize_plan(history, title="Hierarchical Bayes: Blue 'Grow' (3-shot)")
    
    print("\nDemo 完成!")


if __name__ == "__main__":
    demo()
