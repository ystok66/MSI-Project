"""
RSA Notebook Helper - 简化的 Jupyter Notebook 接口

使用方法:
    from rsa_helper import RSAHelper
    
    # 初始化
    rsa = RSAHelper()
    
    # 训练
    rsa.train(["blue box", "", "", ""], "1 blue")
    
    # 推理
    probs = rsa.infer(["blue box", "red box", "", ""], "1 blue")
    
    # 可视化
    rsa.visualize(["blue box", "red solid", "green l", "yellow t"])
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional, Union

# Import core modules
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import infer_posterior, infer_posterior_multi_intent, score_L0, normalize_tokens
from scipy.special import softmax


# =============================================================================
# Color Perturbation
# =============================================================================

def perturb_color(base_rgb: Tuple[int, int, int], 
                  variation: float = 0.15) -> Tuple[int, int, int]:
    """
    对颜色添加随机扰动。
    
    Args:
        base_rgb: 基础 RGB 值 (0-255)
        variation: 扰动比例 (0.0-1.0)
    
    Returns:
        扰动后的 RGB 值
    """
    r, g, b = base_rgb
    
    # 随机扰动
    delta = int(255 * variation)
    new_r = np.clip(r + np.random.randint(-delta, delta + 1), 0, 255)
    new_g = np.clip(g + np.random.randint(-delta, delta + 1), 0, 255)
    new_b = np.clip(b + np.random.randint(-delta, delta + 1), 0, 255)
    
    return (int(new_r), int(new_g), int(new_b))


# =============================================================================
# Scene Parsing
# =============================================================================

def parse_scene_string(scene_list: List[str], 
                       perturb: bool = True,
                       perturbation: float = 0.15) -> Scene:
    """
    解析场景字符串列表。
    
    Args:
        scene_list: ["red box", "blue solid", "", "green l"]
        perturb: 是否添加颜色扰动
        perturbation: 扰动强度
    
    Returns:
        Scene 对象
    """
    regions = []
    
    for slot in scene_list[:4]:  # 最多4个区域
        if not slot or slot.strip() == "":
            regions.append(None)
            continue
        
        # 解析 "color shape"
        parts = slot.strip().lower().split()
        if len(parts) < 2:
            regions.append(None)
            continue
        
        color_name = parts[0]
        shape_name = parts[1]
        
        # 验证颜色和形状
        if color_name not in COLORS_RGB:
            print(f"警告: 未知颜色 '{color_name}', 使用 gray")
            color_name = "gray"
        
        if shape_name not in SHAPES:
            print(f"警告: 未知形状 '{shape_name}', 使用 box")
            shape_name = "box"
        
        # 获取颜色 (可选扰动)
        base_rgb = COLORS_RGB[color_name]
        if perturb:
            rgb = perturb_color(base_rgb, perturbation)
        else:
            rgb = base_rgb
        
        # 创建对象
        occ = np.array(SHAPES[shape_name], dtype=np.float32)
        obj = Obj(shape_name=shape_name, color_rgb=rgb, occ=occ)
        regions.append(obj)
    
    # 填充到4个
    while len(regions) < 4:
        regions.append(None)
    
    return Scene(regions=regions)


def parse_description(desc: str) -> Tuple[List[str], int]:
    """
    解析描述字符串。
    
    Args:
        desc: "1 blue box" 或 "2 red" 或 "blue"
    
    Returns:
        (tokens, count)
        tokens: ["blue", "box"]
        count: 1
    """
    parts = desc.strip().lower().split()
    
    if not parts:
        return [], 1
    
    # 检查第一个是否是数字
    try:
        count = int(parts[0])
        tokens = parts[1:]
    except ValueError:
        count = 1
        tokens = parts
    
    return tokens, count


# =============================================================================
# RSA Helper Class
# =============================================================================

class RSAHelper:
    """
    RSA 语用通信模型的简化接口。
    """
    
    def __init__(self, d: int = 12, alpha: float = 5.0, 
                 color_perturbation: float = 0.15,
                 use_gaussian_shape: bool = True,
                 shape_sigma: float = 0.5,
                 novelty_threshold: float = 3.0,
                 min_kappa_known: float = 2.0):
        """
        初始化 RSA Helper。
        
        Args:
            d: 特征维度 (默认 12: 3 色彩 + 9 形状)
            alpha: RSA 理性参数
            color_perturbation: 颜色扰动强度 (0=无扰动, 0.15=默认)
            use_gaussian_shape: 是否使用高斯平滑形状编码 (默认 True)
            shape_sigma: 高斯平滑的 sigma 值 (默认 0.5)
                         越大越模糊: 0.3=几乎不模糊, 0.5=适中, 1.0=过度模糊
            novelty_threshold: Z-score 阈值，高于此值视为已知 (默认 3.0)
            min_kappa_known: κ 最低要求 (默认 2.0)
        """
        # 设置全局编码参数
        import encoders
        encoders.USE_GAUSSIAN_SHAPE = use_gaussian_shape
        encoders.GAUSSIAN_SIGMA = shape_sigma
        
        self.table = ConceptTable(d=d)
        self.alpha = alpha
        self.perturbation = color_perturbation
        self.use_gaussian_shape = use_gaussian_shape
        self.shape_sigma = shape_sigma
        self.novelty_threshold = novelty_threshold
        self.min_kappa_known = min_kappa_known
        self.training_count = 0
        
        gaussian_info = f", σ_shape={shape_sigma}" if use_gaussian_shape else ", hard_shape"
        print(f"RSA Helper 初始化完成 (d={d}, α={alpha}{gaussian_info})")
    
    def reset(self):
        """重置概念表。"""
        self.table = ConceptTable(d=self.table.d)
        self.training_count = 0
        print("概念表已重置")
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    
    def train(self, scene: List[str], description: str, 
              perturb: bool = None, verbose: bool = True, 
              show: bool = True):
        """
        训练一个场景-描述对。
        
        Args:
            scene: ["blue box", "", "", ""] - 场景描述
            description: "1 blue" - 语言描述
            perturb: 是否添加颜色扰动 (默认使用初始化设置)
            verbose: 是否打印信息
            show: 是否显示训练用的实际颜色
        
        Example:
            rsa.train(["blue box", "", "", ""], "1 blue", show=True)
        """
        if perturb is None:
            perturb = self.perturbation > 0
        
        # 解析场景 (应用颜色扰动)
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        X, mask = encode_scene(scene_obj)
        
        # 解析描述
        tokens, k = parse_description(description)
        
        if not tokens:
            print("错误: 描述不能为空")
            return
        
        # 训练
        learn_step(X, mask, k=k, tokens=tokens, table=self.table)
        self.training_count += 1
        
        if verbose:
            active_slots = [s for s in scene if s.strip()]
            print(f"训练 #{self.training_count}: {active_slots} → \"{description}\"")
        
        # 可视化实际使用的颜色
        if show:
            self._visualize_scene_obj(scene_obj, scene, f"训练 #{self.training_count}: \"{description}\"")
    
    def train_batch(self, examples: List[Tuple[List[str], str]], 
                    verbose: bool = True):
        """
        批量训练。
        
        Args:
            examples: [(scene, description), ...]
        
        Example:
            rsa.train_batch([
                (["blue box", "", "", ""], "1 blue"),
                (["blue solid", "", "", ""], "1 blue"),
                (["red box", "", "", ""], "1 red"),
            ])
        """
        for scene, desc in examples:
            self.train(scene, desc, verbose=verbose)
    
    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    
    def infer(self, scene: List[str], description: str,
              use_rsa: bool = True, perturb: bool = False) -> np.ndarray:
        """
        推理: 根据描述选择场景中的对象。
        
        Args:
            scene: ["blue box", "red box", "", ""] - 测试场景
            description: "1 blue" - 查询描述
            use_rsa: 是否使用 RSA (False = L0)
            perturb: 测试时是否扰动颜色 (默认 False)
        
        Returns:
            4元素概率数组 [P(region0), P(region1), P(region2), P(region3)]
        
        Example:
            probs = rsa.infer(["blue box", "red box", "", ""], "1 blue")
            # probs[0] 接近 1.0
        """
        # 解析场景
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        X, mask = encode_scene(scene_obj)
        
        # 解析描述
        tokens, k = parse_description(description)
        
        if not tokens:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        # 检查是否是多意图
        # 例如 "1 blue, 1 red" -> 多意图
        if "," in description:
            return self._infer_multi_intent(scene, description, use_rsa, perturb)
        
        # 单意图推理
        if use_rsa:
            # 禁用 auto_alt_from_table，避免其他概念干扰推理
            probs = infer_posterior(X, mask, tokens, self.table, 
                                    alpha=self.alpha, auto_alt_from_table=False)
        else:
            # L0
            l0_scores = score_L0(X, mask, normalize_tokens(tokens), self.table)
            probs = softmax(self.alpha * l0_scores)
        
        return probs
    
    def _infer_multi_intent(self, scene: List[str], description: str,
                            use_rsa: bool, perturb: bool) -> np.ndarray:
        """多意图推理的内部方法。"""
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        X, mask = encode_scene(scene_obj)
        
        # 解析多意图: "1 blue, 1 red"
        intents = []
        for part in description.split(","):
            tokens, k = parse_description(part.strip())
            if tokens:
                intents.append((tokens, k))
        
        if not intents:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        # 使用 multi-intent 推理
        result = infer_posterior_multi_intent(
            X, mask, intents, self.table, 
            use_rsa=use_rsa, alpha=self.alpha
        )
        
        # 边际化到单个对象概率
        marginals = np.zeros(4)
        for assignment, prob in result.items():
            for intent_objs in assignment:
                for obj_idx in intent_objs:
                    marginals[obj_idx] += prob
        
        return marginals
    
    def compare(self, scene: List[str], description: str,
                perturb: bool = False) -> Dict[str, np.ndarray]:
        """
        比较 L0 和 RSA 结果。
        
        Returns:
            {"L0": probs, "RSA": probs}
        """
        l0_probs = self.infer(scene, description, use_rsa=False, perturb=perturb)
        rsa_probs = self.infer(scene, description, use_rsa=True, perturb=perturb)
        return {"L0": l0_probs, "RSA": rsa_probs}
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    
    def visualize(self, scene: List[str], title: str = "Scene", 
                  perturb: bool = False) -> plt.Figure:
        """
        可视化场景 (2x2 网格)。
        
        Args:
            scene: ["blue box", "red solid", "green l", ""]
            title: 图表标题
            perturb: 是否扰动颜色
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(4, 4))
        axes = axes.flatten()
        
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        
        for i, (ax, obj_desc) in enumerate(zip(axes, scene[:4])):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f"Region {i}")
            
            if not obj_desc or obj_desc.strip() == "":
                ax.text(0.5, 0.5, "Empty", ha='center', va='center', 
                        fontsize=14, color='gray')
                ax.set_facecolor('#f0f0f0')
            else:
                # 解析并绘制
                parts = obj_desc.strip().lower().split()
                if len(parts) >= 2:
                    color_name, shape_name = parts[0], parts[1]
                    
                    # 获取颜色 (支持扰动)
                    if color_name in COLORS_RGB:
                        rgb = COLORS_RGB[color_name]
                        if perturb:
                            rgb = perturb_color(rgb, self.perturbation)
                        color = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
                    else:
                        color = (0.5, 0.5, 0.5)
                    
                    # 绘制形状
                    if shape_name in SHAPES:
                        self._draw_shape(ax, shape_name, color)
                    else:
                        ax.add_patch(patches.Rectangle((0.2, 0.2), 0.6, 0.6, 
                                                       facecolor=color, edgecolor='black'))
                    
                    ax.set_xlabel(f"{color_name} {shape_name}", fontsize=12)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        return None  # 不返回 Figure 避免 Jupyter 重复显示
    
    def _visualize_scene_obj(self, scene_obj: Scene, scene_strs: List[str], title: str):
        """可视化 Scene 对象，显示实际使用的颜色（包括扰动后的）。"""
        fig, axes = plt.subplots(2, 2, figsize=(4, 4))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
            obj = scene_obj.regions[i] if i < len(scene_obj.regions) else None
            obj_desc = scene_strs[i] if i < len(scene_strs) else ""
            
            if obj is None:
                ax.text(0.5, 0.5, "Empty", ha='center', va='center', 
                        fontsize=10, color='gray')
                ax.set_facecolor('#f0f0f0')
            else:
                # 使用实际的 RGB 颜色（已扰动）
                rgb = obj.color_rgb
                color = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
                self._draw_shape(ax, obj.shape_name, color)
                ax.set_xlabel(f"{obj_desc}\nRGB:{rgb}", fontsize=8)
            
            ax.set_title(f"R{i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    def _draw_shape(self, ax, shape_name: str, color: Tuple[float, float, float]):
        """绘制形状 (基于 3x3 占用矩阵)。"""
        occ = SHAPES[shape_name]
        cell_size = 0.25
        offset = 0.125
        
        for row in range(3):
            for col in range(3):
                if occ[row][col] == 1:
                    x = offset + col * cell_size
                    y = offset + (2 - row) * cell_size  # 翻转 y
                    rect = patches.Rectangle((x, y), cell_size * 0.95, cell_size * 0.95,
                                            facecolor=color, edgecolor='black', 
                                            linewidth=1)
                    ax.add_patch(rect)
    
    def visualize_probs(self, scene: List[str], description: str,
                        show_comparison: bool = True,
                        perturb: bool = False) -> plt.Figure:
        """
        可视化推理概率 (2x2 网格 + 概率)。
        
        Args:
            scene: 场景描述
            description: 查询描述
            show_comparison: 是否同时显示 L0 和 RSA
            perturb: 是否扰动颜色
        
        Returns:
            matplotlib Figure
        """
        results = self.compare(scene, description, perturb)
        l0_probs = results["L0"]
        rsa_probs = results["RSA"]
        
        if show_comparison:
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            
            # L0 结果
            for i in range(4):
                ax = axes[0, i]
                self._draw_prob_cell(ax, scene[i] if i < len(scene) else "", 
                                    l0_probs[i], i)
            axes[0, 0].set_ylabel("L0", fontsize=14, fontweight='bold')
            
            # RSA 结果
            for i in range(4):
                ax = axes[1, i]
                self._draw_prob_cell(ax, scene[i] if i < len(scene) else "", 
                                    rsa_probs[i], i)
            axes[1, 0].set_ylabel("RSA", fontsize=14, fontweight='bold')
            
            plt.suptitle(f'Query: "{description}"', fontsize=16)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            for i, ax in enumerate(axes):
                self._draw_prob_cell(ax, scene[i] if i < len(scene) else "", 
                                    rsa_probs[i], i)
            plt.suptitle(f'Query: "{description}" (RSA)', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        return None
    
    def _draw_prob_cell(self, ax, obj_desc: str, prob: float, idx: int):
        """绘制单个概率格子。"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # 背景色基于概率
        bg_color = (0.2 + 0.6 * prob, 0.8 - 0.4 * prob, 0.2)  # 绿-红渐变
        ax.set_facecolor((*bg_color, 0.3))
        
        if obj_desc and obj_desc.strip():
            parts = obj_desc.strip().lower().split()
            if len(parts) >= 2:
                color_name, shape_name = parts[0], parts[1]
                if color_name in COLORS_RGB and shape_name in SHAPES:
                    rgb = COLORS_RGB[color_name]
                    color = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
                    self._draw_shape(ax, shape_name, color)
            ax.set_title(f"{obj_desc}\nP={prob:.2%}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Empty", ha='center', va='center', 
                   fontsize=10, color='gray')
            ax.set_title(f"Empty\nP={prob:.2%}", fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    def show_grid(self, scene: List[str], description: str,
                  use_rsa: bool = True, perturb: bool = False) -> np.ndarray:
        """
        输出 2x2 概率网格。
        
        Returns:
            2x2 numpy 数组
        """
        probs = self.infer(scene, description, use_rsa=use_rsa, perturb=perturb)
        grid = probs.reshape(2, 2)
        
        print(f"Query: \"{description}\" ({'RSA' if use_rsa else 'L0'})")
        print("=" * 40)
        print(f"  Region 0: {probs[0]:.2%}  |  Region 1: {probs[1]:.2%}")
        print("  " + "-" * 36)
        print(f"  Region 2: {probs[2]:.2%}  |  Region 3: {probs[3]:.2%}")
        print()
        
        return grid
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    def status(self):
        """打印当前状态。"""
        print("=" * 40)
        print("RSA Helper 状态")
        print("=" * 40)
        print(f"训练样本数: {self.training_count}")
        print(f"已学概念: {list(self.table._concepts.keys())}")
        print(f"特征维度: {self.table.d}")
        print(f"α (理性参数): {self.alpha}")
        print(f"颜色扰动: {self.perturbation}")
    
    def concepts(self) -> List[str]:
        """返回已学习的概念列表。"""
        return list(self.table._concepts.keys())
    
    # -------------------------------------------------------------------------
    # Active Learning (主动学习)
    # -------------------------------------------------------------------------
    
    def _ensure_active_learner(self):
        """确保 ActiveLearner 已初始化。"""
        if not hasattr(self, '_active_learner'):
            from active_learner import ActiveLearner
            self._active_learner = ActiveLearner(
                self.table, 
                novelty_threshold=self.novelty_threshold,
                min_kappa_known=self.min_kappa_known,
                verbose=True
            )
        return self._active_learner
    
    def ask(self, scene: List[str], position: int, perturb: bool = False):
        """
        询问模型：位置 `position` 的对象是什么？
        
        如果对象熟悉，返回最佳概念。
        如果对象陌生，生成临时概念。
        
        Args:
            scene: ["blue box", "pink solid", "", ""]
            position: 区域索引 (0-3)
            perturb: 是否扰动颜色
        
        Returns:
            AskResult 对象
        
        Example:
            result = rsa.ask(["blue box", "pink solid", "", ""], position=1)
            print(result.message)
        """
        learner = self._ensure_active_learner()
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        return learner.ask(scene_obj, position)
    
    def answer(self, scene: List[str], position: int, utterance: str, 
               perturb: bool = False):
        """
        处理用户对 Ask 的反馈。
        
        Args:
            scene: 场景描述
            position: 区域索引
            utterance: 用户的描述 (如 "1 pink")
            perturb: 是否扰动颜色
        
        Returns:
            处理结果消息
        
        Example:
            rsa.answer(["blue box", "pink solid", "", ""], position=1, utterance="1 pink")
        """
        learner = self._ensure_active_learner()
        scene_obj = parse_scene_string(scene, perturb, self.perturbation)
        return learner.answer(scene_obj, position, utterance)
    
    def self_train(self, scenes: List[List[str]], perturb: bool = None):
        """
        自监督训练：模型用自己的 Ask 结果作为伪标签。
        
        Args:
            scenes: 场景列表 [["blue box", "", "", ""], [...], ...]
            perturb: 是否扰动颜色 (默认使用初始化设置)
        
        Returns:
            统计: {"known": n, "new": m, "empty": k}
        
        Example:
            stats = rsa.self_train([
                ["blue box", "pink solid", "", ""],
                ["red l", "green t", "", ""]
            ])
        """
        if perturb is None:
            perturb = self.perturbation > 0
        
        learner = self._ensure_active_learner()
        
        # 转换场景
        scene_objs = [parse_scene_string(s, perturb, self.perturbation) for s in scenes]
        
        return learner.self_train(scene_objs)
    
    def reflect(self, z_threshold: float = 1.0, allow_merge_trained: bool = False, 
                verbose: bool = True):
        """
        反思概念表：检测相似概念并合并。
        
        基于 KL 散度的 Z-score 判断概念相似性。
        
        合并规则：
        1. 临时概念 + 训练概念 → 用训练概念名
        2. 两个训练概念 → 
           - allow_merge_trained=False: 不合并
           - allow_merge_trained=True: 合并，名字变成 frozenset({'red', 'blue'})
        3. 两个临时概念 → 随机选一个
        
        Args:
            z_threshold: Z-score 阈值 (默认 1.0)
            allow_merge_trained: 是否允许合并两个训练概念 (默认 False)
            verbose: 是否打印详细信息
        
        Returns:
            统计信息: {
                "pairs_checked": int,
                "merges": List[Tuple[str, str, str]],
                "skipped_both_trained": int
            }
        
        Example:
            # 不合并训练概念
            result = rsa.reflect(z_threshold=1.0)
            
            # 允许合并训练概念 (名字变成 frozenset)
            result = rsa.reflect(z_threshold=1.0, allow_merge_trained=True)
            print(f"合并了 {len(result['merges'])} 对概念")
        """
        learner = self._ensure_active_learner()
        return learner.reflect(z_threshold=z_threshold, 
                              allow_merge_trained=allow_merge_trained,
                              verbose=verbose)
    
    # -------------------------------------------------------------------------
    # Zero-Shot Learning (零样本学习)
    # -------------------------------------------------------------------------
    
    def add_embedding(self, token: str, embedding: np.ndarray) -> bool:
        """
        给已知概念注入语义向量 (用于 Zero-Shot Learning)。
        
        Args:
            token: 概念名称 (必须已存在于概念表)
            embedding: 语义向量 (如 Word2Vec/GloVe/CLIP)
            
        Returns:
            True 如果成功注入, False 如果概念不存在
            
        Example:
            # 给已学的 'red' 概念添加语义向量
            rsa.add_embedding("red", np.array([1.0, 0.0, 0.0]))
        """
        return self.table.add_embedding(token, embedding)
    
    def synthesize_concept(self, new_token: str, new_embedding: np.ndarray,
                          temp: float = 0.1, uncertainty_scale: float = 1.2,
                          verbose: bool = True):
        """
        Zero-Shot Learning：根据语义相似度，利用已知概念合成一个新概念。
        
        核心假设：语言空间的几何结构与感知空间的几何结构是同构的。
        如果 Vec(Orange) ≈ 0.5 * Vec(Red) + 0.5 * Vec(Yellow)，
        那么 Mu(Orange) ≈ 0.5 * Mu(Red) + 0.5 * Mu(Yellow)。
        
        Args:
            new_token: 新概念名称 (如 "orange")
            new_embedding: 新概念的语义向量
            temp: 温度系数
                  0.01 = 只关注最相似的词 (Nearest Neighbor)
                  0.1  = 关注相似的几个词 (Interpolation)
                  1.0  = 所有词平均 (Average)
            uncertainty_scale: 方差放大系数 (默认 1.2)
            verbose: 是否打印详细信息
            
        Returns:
            合成的 Concept 对象，如果失败返回 None
            
        Example:
            # 假设已学过 red, yellow, blue 并添加了 embedding
            rsa.train([["red box"]], "1 red")
            rsa.add_embedding("red", np.array([1.0, 0.0, 0.0]))
            rsa.add_embedding("yellow", np.array([0.0, 1.0, 0.0]))
            
            # 零样本合成 orange
            vec_orange = np.array([0.7, 0.7, 0.0])
            rsa.synthesize_concept("orange", vec_orange)
            
            # 现在可以直接用 'orange' 推理了!
            probs = rsa.infer(["orange box", "red box", "", ""], "1 orange")
        """
        return self.table.synthesize_concept(
            new_token, new_embedding, 
            temp=temp, 
            uncertainty_scale=uncertainty_scale, 
            verbose=verbose
        )
    
    def synthesize_concept_gp(self, new_token: str, new_embedding: np.ndarray,
                              length_scale: float = 0.5,
                              signal_var: float = 1.0,
                              noise_var: float = 0.01,
                              var_floor: float = 0.01,
                              verbose: bool = True):
        """
        Zero-Shot Learning (GP版)：用高斯过程回归合成新概念。
        
        相比 synthesize_concept (softmax 核插值)，GP 版本提供严格的
        epistemic + aleatoric 不确定性分解：
            σ²_new = s²_epi(v*) + σ²_ale(v*) + ε_min
        
        Args:
            new_token: 新概念名称 (如 "orange")
            new_embedding: 新概念的语义向量
            length_scale: RBF 核长度尺度 (默认 0.5)
            signal_var: 信号方差 σ²_f
            noise_var: 观测噪声 σ²_n  
            var_floor: 最小方差 ε_min
            verbose: 是否打印详细信息
            
        Returns:
            合成的 Concept 对象，如果失败返回 None
            
        Example:
            rsa.add_embedding("red", np.array([1.0, 0.0, 0.0]))
            rsa.add_embedding("yellow", np.array([0.0, 1.0, 0.0]))
            
            # GP 合成 orange — 自动获得 epistemic 不确定性
            rsa.synthesize_concept_gp("orange", np.array([0.7, 0.7, 0.0]))
        """
        return self.table.synthesize_concept_gp(
            new_token, new_embedding,
            length_scale=length_scale,
            signal_var=signal_var,
            noise_var=noise_var,
            var_floor=var_floor,
            verbose=verbose
        )
    
    def grounded_concepts(self) -> list:
        """返回所有有 embedding 的概念名称列表。"""
        return self.table.grounded_concepts()
    
    # -------------------------------------------------------------------------
    # Memory Decay (Jost's Law)
    # -------------------------------------------------------------------------
    
    def sleep(self, base_rate: float = 0.3, stability: float = 1.0,
              prune_threshold: float = 50.0, verbose: bool = True) -> dict:
        """
        睡眠/记忆整理：模拟 Jost's Law 记忆衰减。
        
        建议在每个 Episode/Batch 结束后调用，模拟"一天结束后的记忆整理"。
        
        核心机制：
        - 新概念 (刚学的) 衰减快，容易遗忘
        - 老概念 (反复复习的) 衰减慢，稳如磐石
        - 衰减率: λ(t) = α / (1 + β * t)
        
        衰减方式：
        - 精度丢失 (Blurring): var 增大，记忆变模糊
        - κ 不衰减: 观测计数是事实，不会被遗忘
        - 当方差膨胀到阈值以上时，概念被彻底遗忘
        
        Args:
            base_rate (α): 初始衰减率 (0.0~1.0)
                          0.3 = 新概念每次衰减 30% (默认)
                          0.5 = 激进遗忘
                          0.1 = 温和遗忘
            stability (β): 稳固系数
                          1.0 = 标准 (默认)
                          2.0 = 概念更快进入"长时记忆"
            prune_threshold: 平均方差超过此值时概念被彻底删除 (默认 50.0)
            verbose: 是否打印详细信息
            
        Returns:
            统计: {"decayed": n, "pruned": m, "survivors": k}
            
        Example:
            # 训练一批数据
            rsa.train(...)
            rsa.self_train(...)
            
            # 睡觉整理记忆
            rsa.sleep(base_rate=0.3)
            
            # 噪音概念会被遗忘，核心概念会保留
        """
        return self.table.apply_memory_decay(
            base_rate=base_rate,
            stability=stability,
            prune_threshold=prune_threshold,
            verbose=verbose
        )
# =============================================================================
# Quick Demo
# =============================================================================

def demo():
    """快速演示。"""
    print("=" * 60)
    print("RSA Helper Demo")
    print("=" * 60)
    
    # 初始化
    rsa = RSAHelper(color_perturbation=0.1)
    
    # 训练
    print("\n--- 训练 ---")
    rsa.train_batch([
        (["blue box", "", "", ""], "1 blue"),
        (["blue solid", "", "", ""], "1 blue"),
        (["blue l", "", "", ""], "1 blue"),
        (["red box", "", "", ""], "1 red"),
        (["red solid", "", "", ""], "1 red"),
    ])
    
    # 状态
    print("\n--- 状态 ---")
    rsa.status()
    
    # 推理
    print("\n--- 推理测试 ---")
    scene = ["blue box", "red box", "", ""]
    
    # L0 vs RSA
    results = rsa.compare(scene, "1 blue")
    print(f"Scene: {scene}")
    print(f"Query: '1 blue'")
    print(f"L0:  {[f'{p:.2%}' for p in results['L0']]}")
    print(f"RSA: {[f'{p:.2%}' for p in results['RSA']]}")
    
    # 2x2 网格
    print("\n--- 2x2 概率网格 ---")
    rsa.show_grid(scene, "1 blue")
    
    print("\nDemo 完成!")


if __name__ == "__main__":
    demo()
