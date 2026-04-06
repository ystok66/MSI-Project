"""
Concept representation for RSA research project.

Provides:
- Concept: Single token's learned parameters (mu, var)
- ConceptTable: Collection of concepts with lazy initialization
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class Concept:
    """
    A single concept (word/token) with Gaussian parameters.
    
    Attributes:
        token: The word/token string (lowercase)
        mu: Mean vector, shape (d,)
        var: Diagonal variance vector, shape (d,)
        kappa: Observation count for online learning (Step 4)
        embedding: Optional semantic embedding vector for Zero-Shot Learning
        decay_count: Memory age counter for Jost's Law decay (0 = new, fragile)
    """
    token: str
    mu: np.ndarray
    var: np.ndarray
    kappa: float = 0.0
    embedding: Optional[np.ndarray] = None
    decay_count: int = 0  # 记忆年龄：经历过多少次衰减周期
    
    def __post_init__(self):
        # Ensure arrays are float64
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.var = np.asarray(self.var, dtype=np.float64)
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float64)


class ConceptTable:
    """
    Collection of concepts with lazy initialization.
    
    New tokens are automatically initialized with a weak prior
    when accessed via ensure().
    
    Attributes:
        d: Feature dimension (12 for Lab+shape)
        mu0: Prior mean for new concepts
        var0: Prior variance for new concepts
        kappa0: Initial pseudo-count for new concepts
    """
    
    def __init__(
        self,
        d: int,
        mu0: Optional[np.ndarray] = None,
        var0: Optional[np.ndarray] = None,
        kappa0: float = 0.5,
        use_centered_prior: bool = False
    ):
        """
        Initialize concept table.
        
        Args:
            d: Feature dimension
            mu0: Prior mean (default: zeros, or centered if use_centered_prior=True)
            var0: Prior variance (default: ones)
            kappa0: Initial pseudo-count
            use_centered_prior: If True, use empirically centered prior for more uniform L0
        """
        self.d = d
        
        if mu0 is not None:
            self.mu0 = mu0
        elif use_centered_prior and d == 12:
            # Empirically centered prior for d=12 (Lab color + shape)
            # Makes L0 more uniform for new tokens
            self.mu0 = np.array([
                0.23, 0.06, 0.21,  # Lab color center
                1.0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75  # shape center
            ], dtype=np.float64)
        else:
            self.mu0 = np.zeros(d, dtype=np.float64)
        
        # var0=1.0 allows learned concepts to develop tight discrimination
        self.var0 = var0 if var0 is not None else np.ones(d, dtype=np.float64)
        self.kappa0 = kappa0
        
        self._concepts: Dict[str, Concept] = {}
    
    def ensure(self, token: str) -> Concept:
        """
        Get concept for token, creating it with prior if not exists.
        
        This is the primary access method - safe for zero-shot tokens.
        
        Args:
            token: Word/token string (will be lowercased)
            
        Returns:
            Concept for the token
        """
        token = token.lower().strip()
        
        if token not in self._concepts:
            self._concepts[token] = Concept(
                token=token,
                mu=self.mu0.copy(),
                var=self.var0.copy(),
                kappa=self.kappa0
            )
        
        return self._concepts[token]
    
    def get(self, token: str) -> Concept:
        """
        Get concept for token, raising KeyError if not exists.
        
        Use ensure() instead for safe access.
        
        Args:
            token: Word/token string
            
        Returns:
            Concept for the token
            
        Raises:
            KeyError: If token not in table
        """
        token = token.lower().strip()
        return self._concepts[token]
    
    def has(self, token: str) -> bool:
        """Check if token exists in table."""
        return token.lower().strip() in self._concepts
    
    def __len__(self) -> int:
        """Number of concepts in table."""
        return len(self._concepts)
    
    def tokens(self) -> list:
        """List of all tokens in table."""
        return list(self._concepts.keys())
    
    def add_embedding(self, token: str, embedding: np.ndarray) -> bool:
        """
        给已知概念注入语义向量 (用于 Zero-Shot Learning)。
        
        Args:
            token: 概念名称
            embedding: 语义向量 (如 Word2Vec/GloVe/CLIP)
            
        Returns:
            True 如果成功注入, False 如果概念不存在
        """
        token = token.lower().strip()
        
        # 归一化向量，方便后续算余弦相似度
        embedding = np.asarray(embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        if token in self._concepts:
            self._concepts[token].embedding = embedding
            return True
        else:
            return False
    
    def synthesize_concept(self, new_token: str, new_embedding: np.ndarray,
                          temp: float = 0.1, uncertainty_scale: float = 1.2,
                          verbose: bool = False) -> Optional[Concept]:
        """
        Zero-Shot Learning：根据语义相似度，利用已知概念合成一个新概念。
        
        核心思路：利用嵌入空间的几何关系（同构性假设），
        直接"合成"一个新概念的视觉表示（高斯分布）。
        
        Args:
            new_token: 新概念名称
            new_embedding: 新概念的语义向量
            temp: 温度系数
                  0.01 = 只关注最相似的词 (Nearest Neighbor)
                  0.1  = 关注相似的几个词 (Interpolation)
                  1.0  = 所有词平均 (Average)
            uncertainty_scale: 方差放大系数，表示零样本的不确定性 (默认 1.2)
            verbose: 是否打印详细信息
            
        Returns:
            合成的 Concept 对象，如果失败返回 None
        """
        new_token = new_token.lower().strip()
        
        # 1. 准备数据：找出所有"既有视觉概念又有Embedding"的锚点
        anchors = [c for c in self._concepts.values() if c.embedding is not None]
        if not anchors:
            if verbose:
                print("[synthesize_concept] 失败: 没有找到带有 embedding 的概念")
            return None
        
        # 归一化新向量
        new_embedding = np.asarray(new_embedding, dtype=np.float64)
        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm
        
        # 2. 计算语义权重 (Attention Weights)
        sims = []
        for c in anchors:
            # Cosine Similarity
            sim = np.dot(new_embedding, c.embedding)
            sims.append(sim)
        
        sims = np.array(sims)
        
        # Softmax 归一化 (技巧：减去最大值防止溢出)
        exp_sims = np.exp((sims - np.max(sims)) / temp)
        weights = exp_sims / np.sum(exp_sims)
        
        # 打印权重供调试
        if verbose:
            print(f"[synthesize_concept] 合成 '{new_token}':")
            for c, w in zip(anchors, weights):
                if w > 0.01:
                    print(f"  - {w:.1%} from {c.token}")
        
        # 3. 视觉特征合成 (Linear Interpolation)
        # 假设 encoder 使用了 Lab 颜色空间，它是感知线性的，所以线性插值非常有效
        mixed_mu = np.zeros_like(anchors[0].mu)
        mixed_var = np.zeros_like(anchors[0].var)
        
        for c, w in zip(anchors, weights):
            mixed_mu += w * c.mu
            # 线性插值方差：保持分布的"紧致性"
            mixed_var += w * c.var
        
        # 4. 增加不确定性
        # 因为是猜出来的，方差要比直接学到的稍微大一点
        mixed_var *= uncertainty_scale
        
        # 5. 创建新概念
        new_concept = Concept(
            token=new_token,
            mu=mixed_mu,
            var=mixed_var,
            kappa=0.1,  # 虚拟样本数设得很低，表示这是弱先验
            embedding=new_embedding
        )
        
        # 加入表，这样 RSA 就能用了！
        self._concepts[new_token] = new_concept
        
        if verbose:
            print(f"[synthesize_concept] 成功创建 '{new_token}' (kappa=0.1)")
        
        return new_concept
    
    def synthesize_concept_gp(self, new_token: str, new_embedding: np.ndarray,
                              length_scale: float = 0.5,
                              signal_var: float = 1.0,
                              noise_var: float = 0.01,
                              var_floor: float = 0.01,
                              verbose: bool = False) -> Optional[Concept]:
        """
        Zero-Shot Learning (GP版)：用高斯过程回归合成新概念。
        
        相比 synthesize_concept (softmax 核插值)，GP 版本提供：
        - 严格的 epistemic 不确定性 (外推时方差自动增大)
        - aleatoric 不确定性 (锚点自身方差的加权插值)
        - 二者可分解: σ²_new = s²_epi + σ²_ale + ε_min
        
        每个感知维度一个独立 GP，共享核超参。
        
        Args:
            new_token: 新概念名称
            new_embedding: 新概念的语义向量
            length_scale: RBF 核长度尺度 l
                0.3 = 窄核 (只看最相似的锚点)
                0.5 = 默认
                1.0 = 宽核 (平滑插值)
            signal_var: 信号方差 σ²_f (核的幅度)
            noise_var: 观测噪声 σ²_n (正则化 / jitter)
            var_floor: 最小方差 ε_min
            verbose: 是否打印详细信息
            
        Returns:
            合成的 Concept 对象，如果失败返回 None
        """
        new_token = new_token.lower().strip()
        
        # 1. 准备锚点
        anchors = [c for c in self._concepts.values() if c.embedding is not None]
        if not anchors:
            if verbose:
                print("[synthesize_concept_gp] 失败: 没有带 embedding 的概念")
            return None
        
        n = len(anchors)
        d = self.d
        
        # 归一化新向量
        new_embedding = np.asarray(new_embedding, dtype=np.float64)
        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm
        
        # 2. 构建训练数据
        V = np.array([c.embedding for c in anchors])  # (n, e)
        Y = np.array([c.mu for c in anchors])          # (n, d)
        anchor_vars = np.array([c.var for c in anchors])  # (n, d)
        
        # 3. RBF 核矩阵
        # K_ij = σ²_f * exp(-||v_i - v_j||² / (2l²))
        sq_dists = np.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=2)  # (n, n)
        K = signal_var * np.exp(-sq_dists / (2.0 * length_scale ** 2))
        K_noise = K + noise_var * np.eye(n)  # (K + σ²_n I)
        
        # k*: 新点与训练点的核向量
        sq_dists_star = np.sum((V - new_embedding[None, :]) ** 2, axis=1)  # (n,)
        k_star = signal_var * np.exp(-sq_dists_star / (2.0 * length_scale ** 2))  # (n,)
        
        # k**: 新点的自核
        k_star_star = signal_var  # k(v*, v*) = σ²_f
        
        # 4. GP 预测
        # α = (K + σ²_n I)⁻¹ k*
        try:
            L = np.linalg.cholesky(K_noise)  # Cholesky 分解 (更稳定)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, k_star))  # (n,)
        except np.linalg.LinAlgError:
            # Cholesky 失败，回退到直接求逆
            alpha = np.linalg.solve(K_noise, k_star)  # (n,)
        
        # 预测均值: m* = k*ᵀ (K + σ²_n I)⁻¹ Y = αᵀ Y
        pred_mu = alpha @ Y  # (d,)
        
        # 预测方差 (epistemic, 标量, 所有维度共享):
        # s²_epi = k** - k*ᵀ (K + σ²_n I)⁻¹ k* = k** - αᵀ k*
        s_epi_sq = max(k_star_star - alpha @ k_star, 0.0)
        
        # 5. Aleatoric 不确定性 (GP-weighted 锚点方差插值)
        # GP 权重 α 可能有负值 (这正是 GP 的特点)
        # 用 |α| 归一化后插值方差
        abs_alpha = np.abs(alpha)
        w_ale = abs_alpha / (abs_alpha.sum() + 1e-12)
        pred_ale_var = w_ale @ anchor_vars  # (d,)
        
        # 6. 组装最终方差: σ²_new = s²_epi + σ²_ale + ε_min
        pred_var = s_epi_sq + pred_ale_var + var_floor
        pred_var = np.maximum(pred_var, var_floor)
        
        # 7. 创建概念
        new_concept = Concept(
            token=new_token,
            mu=pred_mu,
            var=pred_var,
            kappa=0.1,  # 弱先验
            embedding=new_embedding
        )
        
        self._concepts[new_token] = new_concept
        
        if verbose:
            print(f"[synthesize_concept_gp] 合成 '{new_token}':")
            print(f"  锚点数: {n}")
            print(f"  GP epistemic σ²: {s_epi_sq:.4f}")
            print(f"  Aleatoric σ² (mean): {np.mean(pred_ale_var):.4f}")
            print(f"  Total σ² (mean): {np.mean(pred_var):.4f}")
            # 打印 GP 权重
            for c, a in zip(anchors, alpha):
                if abs(a) > 0.01:
                    print(f"  - {a:+.3f} × {c.token}")
        
        return new_concept
    
    def grounded_concepts(self) -> list:
        """返回所有有 embedding 的概念名称列表。"""
        return [c.token for c in self._concepts.values() if c.embedding is not None]
    
    def apply_memory_decay(self, base_rate: float = 0.3, stability: float = 1.0,
                           prune_threshold: float = 50.0, verbose: bool = True) -> dict:
        """
        记忆衰减机制：模拟 Jost's Law 遗忘曲线与记忆巩固。
        
        核心原理：
        - 新概念 (decay_count=0) 衰减剧烈，容易被遗忘
        - 老概念 (decay_count 高) 衰减缓慢，稳如磐石
        - 衰减率: λ(t) = α / (1 + β * t)
        
        衰减方式：
        - 精度丢失 (Blurring): var 增大，记忆变模糊
        - κ 不衰减: κ 是观测计数（"见过多少次"），不会被遗忘
        - 当方差膨胀到阈值以上时，概念被彻底遗忘 (pruning)
        
        Args:
            base_rate (α): 初始衰减率 (0.0~1.0)，新概念的遗忘速度
            stability (β): 稳固系数，越大则概念越快进入"长时记忆"
            prune_threshold: 平均方差超过此值时概念被彻底删除
                            (默认 50.0，方差初始为 1.0)
            verbose: 是否打印详细信息
            
        Returns:
            统计信息: {"decayed": n, "pruned": m, "survivors": k}
        """
        tokens_to_remove = []
        stats = {"decayed": 0, "pruned": 0, "survivors": 0}
        
        if verbose:
            print(f"\n--- 🌙 Memory Decay Cycle (α={base_rate}, β={stability}) ---")
        
        for token, concept in list(self._concepts.items()):
            # 1. 计算当前的衰减率 (Jost's Law)
            # count=0 (新) -> rate = α (剧烈遗忘)
            # count=50 (老) -> rate ≈ 0 (坚如磐石)
            rate = base_rate / (1.0 + stability * concept.decay_count)
            
            # 2. 精度丢失 (Blurring) - 模拟细节模糊
            #    κ 不衰减 — 观测计数是事实，不会遗忘
            old_var_mean = np.mean(concept.var)
            concept.var *= (1.0 + rate)
            
            # 3. 增加年龄计数 (Consolidation)
            concept.decay_count += 1
            
            stats["decayed"] += 1
            
            # 4. 彻底遗忘检查 (Pruning)
            #    当方差膨胀到阈值以上 → 概念已模糊到无用
            avg_var = np.mean(concept.var)
            if avg_var > prune_threshold:
                tokens_to_remove.append(token)
                if verbose:
                    print(f"   🗑️ Pruned '{token}' (avg_var={avg_var:.1f} > {prune_threshold})")
            elif rate > 0.05 and verbose:
                # 只打印显著衰减的概念
                print(f"   📉 Blurring '{token}': rate={rate:.2%}, "
                      f"var: {old_var_mean:.2f}→{avg_var:.2f}, κ={concept.kappa:.1f}, age={concept.decay_count}")
        
        # 执行物理删除
        for t in tokens_to_remove:
            del self._concepts[t]
        
        stats["pruned"] = len(tokens_to_remove)
        stats["survivors"] = len(self._concepts)
        
        if verbose:
            print(f"--- Decay complete: {stats['decayed']} decayed, "
                  f"{stats['pruned']} pruned, {stats['survivors']} survivors ---\n")
        
        return stats
