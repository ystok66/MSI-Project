"""
Active Learning Module - 主动学习与元认知

实现功能:
1. Ask (查询与内省): 判断对象是否已知，未知则生成临时概念
2. Answer (交互反馈): 处理用户反馈，确认/重命名/纠正概念
3. Self-Train (自监督): 模型用自己的判断作为伪标签训练

使用方法:
    from active_learner import ActiveLearner
    
    learner = ActiveLearner(table)
    
    # Ask
    response, token = learner.ask(scene, position=0)
    
    # Answer
    learner.answer(scene, position=0, utterance="1 pink")
    
    # Self-Train
    learner.self_train([scene1, scene2, ...])
"""

import numpy as np
import uuid
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from concepts import ConceptTable
from world import Scene, Obj
from encoders import encode_scene
from scoring import log_inc_single
from learner import learn_step


@dataclass
class AskResult:
    """Ask 查询结果"""
    is_known: bool              # 是否是已知概念
    best_token: Optional[str]   # 最佳匹配的概念 (可能为 None)
    best_score: float           # 最佳匹配分数
    familiarity: float          # 熟知度分数
    provisional_token: Optional[str]  # 临时生成的概念 (如果未知)
    message: str                # 人类可读的响应


class ActiveLearner:
    """
    主动学习器 - 支持元认知（知道自己知道什么/不知道什么）
    """
    
    def __init__(self, 
                 table: ConceptTable,
                 novelty_threshold: float = 3.0,
                 min_kappa_known: float = 2.0,
                 verbose: bool = True):
        """
        初始化主动学习器。
        
        Args:
            table: 概念表
            novelty_threshold: Z-score 阈值，高于此值视为已知 (默认 3.0)
            min_kappa_known: κ 最低要求，低于此值即使 Z-score 高也判为未知 (默认 2.0)
            verbose: 是否打印详细信息
        """
        self.table = table
        self.novelty_threshold = novelty_threshold
        self.min_kappa_known = min_kappa_known
        self.verbose = verbose
        
        # 临时概念注册表: (scene_id, region_idx) -> provisional_token
        self.pending_concepts: Dict[Tuple[int, int], str] = {}
    
    # =========================================================================
    # Domain-Aware Z-score Novelty Detection (领域感知的Z分数新颖性检测)
    # =========================================================================
    
    def _sample_valid_pseudo_object(self) -> np.ndarray:
        """
        生成合法的随机物体嵌入 (Domain-Aware Sampling)。
        
        关键：从模板中随机选择颜色和形状，生成真实的物体嵌入。
        这样基准线才是"随便拿一个真实物体"的分数。
        
        Returns:
            合法的 d 维特征向量 [L, a, b, occ_0, ..., occ_8]
        """
        from templates import SHAPES, COLORS_RGB
        from world import Obj, Scene
        from encoders import encode_scene
        
        # 随机选择一个真实的颜色和形状
        color = np.random.choice(list(COLORS_RGB.keys()))
        shape = np.random.choice(list(SHAPES.keys()))
        
        # 创建真实的物体并编码
        obj = Obj(
            shape_name=shape,
            color_rgb=COLORS_RGB[color],
            occ=np.array(SHAPES[shape], dtype=np.float32)
        )
        scene = Scene(regions=[obj, None, None, None])
        X, _ = encode_scene(scene)
        
        return X[0]
    
    def update_background_stats(self, n_samples: int = 50):
        """
        更新背景统计量 (应在 learn_step 后调用或懒加载)。
        
        计算随机合法物体在当前概念表下的得分分布。
        """
        if not self.table._concepts:
            self._bg_stats = (-100.0, 1.0)  # 默认值
            return
        
        random_scores = []
        for _ in range(n_samples):
            x_rand = self._sample_valid_pseudo_object()
            
            # 计算这个随机物体在当前概念表里的最大得分
            max_score = -float('inf')
            for token, concept in self.table._concepts.items():
                score = log_inc_single(x_rand, concept.mu, concept.var)
                if score > max_score:
                    max_score = score
            
            random_scores.append(max_score)
        
        mu = np.mean(random_scores)
        sigma = np.std(random_scores) + 1e-6
        self._bg_stats = (mu, sigma)
    
    def familiarity_score(self, x: np.ndarray, 
                          n_baseline_samples: int = 100) -> Tuple[float, Optional[str], float, dict]:
        """
        基于 Domain-Aware Z-score 计算熟知度。
        
        双重检查:
        1. 区分度: Z-score = (obj_score - μ_bg) / σ_bg > novelty_threshold
        2. 证据强度: κ >= min_kappa_known
        两者同时满足才判为已知。
        
        Args:
            x: 对象的特征向量 (d,)
            n_baseline_samples: 背景采样数量 (默认 100)
        
        Returns:
            (z_score, best_token, best_score, detail_dict)
            - z_score: 标准化显著性分数
            - detail_dict: {"mu_bg", "sigma_bg", "kappa", "is_known"}
        """
        if not self.table._concepts:
            return 0.0, None, -float('inf'), {"is_known": False, "kappa": 0, "mu_bg": 0, "sigma_bg": 0}
        
        # 1. 计算当前对象的最大分数
        best_score = -float('inf')
        best_token = None
        best_kappa = 0.0
        
        for token, concept in self.table._concepts.items():
            score = log_inc_single(x, concept.mu, concept.var)
            if score > best_score:
                best_score = score
                best_token = token
                best_kappa = getattr(concept, 'kappa', 1.0)
        
        # 2. 获取或更新背景统计量
        if not hasattr(self, '_bg_stats') or self._bg_stats is None:
            self.update_background_stats(n_baseline_samples)
        mu_bg, sigma_bg = self._bg_stats
        
        # 3. 计算 Z-score
        z_score = (best_score - mu_bg) / sigma_bg
        
        # 4. Kappa 加权
        kappa_weight = np.log10(best_kappa + 1)
        
        # 5. 双重判定: Z-score 超过阈值 AND κ 证据充足
        is_known = (z_score > self.novelty_threshold) and (best_kappa >= self.min_kappa_known)
        
        detail = {
            "mu_bg": mu_bg,
            "sigma_bg": sigma_bg,
            "kappa": best_kappa,
            "kappa_weight": kappa_weight,
            "is_known": is_known
        }
        
        return z_score, best_token, best_score, detail

    # =========================================================================
    # Ask (查询与内省)
    # =========================================================================
    
    def ask(self, scene: Scene, position: int) -> AskResult:
        """
        询问模型：位置 `position` 的对象是什么？
        
        使用双重检查判定：
        - Z > novelty_threshold (默认 3.0) 且 κ >= min_kappa_known (默认 2.0) → 已知
        - 否则 → 未知
        
        Args:
            scene: 场景对象
            position: 区域索引 (0-3)
        
        Returns:
            AskResult 包含判断结果
        """
        # 检查位置是否有效
        if position >= len(scene.regions) or scene.regions[position] is None:
            return AskResult(
                is_known=False,
                best_token=None,
                best_score=-float('inf'),
                familiarity=0.0,
                provisional_token=None,
                message="这里是空的。"
            )
        
        # 获取对象特征
        X, mask = encode_scene(scene)
        x = X[position]
        
        # 计算 Z-score 熟知度
        z_score, best_token, best_score, detail = self.familiarity_score(x)
        is_known = detail["is_known"]
        
        if not is_known:
            # === 未知对象 ===
            # 生成临时概念
            provisional_token = "concept_" + str(uuid.uuid4())[:5]
            
            # 注册到临时表
            scene_id = id(scene)
            self.pending_concepts[(scene_id, position)] = provisional_token
            
            # 构建响应
            fallback = best_token if best_token else "unknown"
            message = (f"我觉得应该用一个新词 '{provisional_token}'。"
                      f"如果硬要问我，最接近的是 '{fallback}' "
                      f"(Z={z_score:.2f}, κ={detail['kappa']:.0f})。")
            
            if self.verbose:
                print(f"[Ask] Position {position}: UNKNOWN (Z={z_score:.2f}, κ={detail['kappa']:.0f})")
                print(f"      Generated provisional: '{provisional_token}'")
            
            return AskResult(
                is_known=False,
                best_token=best_token,
                best_score=best_score,
                familiarity=z_score,
                provisional_token=provisional_token,
                message=message
            )
        else:
            # === 已知对象 ===
            message = (f"我觉得是 '{best_token}' "
                      f"(Z={z_score:.2f}, κ={detail['kappa']:.0f})。")
            
            if self.verbose:
                print(f"[Ask] Position {position}: KNOWN '{best_token}' "
                      f"(Z={z_score:.2f}, κ={detail['kappa']:.0f})")
            
            return AskResult(
                is_known=True,
                best_token=best_token,
                best_score=best_score,
                familiarity=z_score,
                provisional_token=None,
                message=message
            )
    
    # =========================================================================
    # Answer (交互反馈)
    # =========================================================================
    
    def answer(self, scene: Scene, position: int, utterance: str) -> str:
        """
        处理用户对 Ask 的反馈。
        
        支持三种情况:
        1. 确认已知概念: "1 red" (red 在 table 中)
        2. 确认/重命名临时概念: "1 pink" (pink 不在 table 中)
        3. 用已知概念替代临时概念: Ask 说是 concept_xxxxx, 用户说 "1 red"
        
        Args:
            scene: 场景对象
            position: 区域索引
            utterance: 用户的描述 (如 "1 red", "1 pink")
        
        Returns:
            处理结果消息
        """
        # 解析 utterance
        tokens = self._parse_utterance(utterance)
        if not tokens:
            return "无法解析描述。"
        
        obj = scene.regions[position]
        if obj is None:
            return "该位置为空。"
        
        scene_id = id(scene)
        pending_key = (scene_id, position)
        pending_token = self.pending_concepts.get(pending_key)
        
        # 获取特征
        X, full_mask = encode_scene(scene)
        
        # 关键修复：只用当前位置的 mask，避免其他对象的特征污染概念
        single_mask = np.zeros(4, dtype=bool)
        single_mask[position] = True
        
        results = []
        
        for token in tokens:
            if token in self.table._concepts:
                # === Case A: 用户回答了已知概念 ===
                learn_step(X, single_mask, k=1, tokens=[token], table=self.table)
                results.append(f"巩固已知概念 '{token}'")
                
                # 如果有临时概念，取消它
                if pending_token:
                    del self.pending_concepts[pending_key]
                    results.append(f"取消临时概念 '{pending_token}'")
                    pending_token = None
                    
            else:
                # === Case B: 用户回答了新概念 ===
                if pending_token:
                    # 用新名字替换临时概念
                    learn_step(X, single_mask, k=1, tokens=[token], table=self.table)
                    del self.pending_concepts[pending_key]
                    results.append(f"用 '{token}' 替换临时概念 '{pending_token}'")
                    pending_token = None
                else:
                    # 直接创建新概念
                    learn_step(X, single_mask, k=1, tokens=[token], table=self.table)
                    results.append(f"创建新概念 '{token}'")
        
        message = "；".join(results) + "。"
        
        # 清除背景统计缓存，下次 ask 时会重新计算
        self._bg_stats = None
        
        if self.verbose:
            print(f"[Answer] Position {position}: {message}")
        
        return message
    
    def _parse_utterance(self, utterance: str) -> List[str]:
        """解析 utterance，提取概念词。"""
        parts = utterance.strip().lower().split()
        if not parts:
            return []
        
        # 跳过数字
        try:
            int(parts[0])
            return parts[1:]
        except ValueError:
            return parts
    
    # =========================================================================
    # Self-Train (自监督学习)
    # =========================================================================
    
    def self_train(self, scenes: List[Scene], verbose: bool = None) -> Dict[str, int]:
        """
        自监督训练模式：模型用自己的 Ask 结果作为伪标签。
        
        Args:
            scenes: 场景列表
            verbose: 是否打印详细信息 (默认使用 self.verbose)
        
        Returns:
            统计: {"known": n, "new": m, "empty": k}
        """
        if verbose is None:
            verbose = self.verbose
        
        stats = {"known": 0, "new": 0, "empty": 0}
        
        for scene_idx, scene in enumerate(scenes):
            for pos in range(len(scene.regions)):
                if scene.regions[pos] is None:
                    stats["empty"] += 1
                    continue
                
                # 1. Ask Myself
                result = self.ask(scene, pos)
                
                # 2. Determine token to use
                if result.is_known:
                    token = result.best_token
                    stats["known"] += 1
                else:
                    token = result.provisional_token
                    stats["new"] += 1
                
                # 3. Answer Myself (train with pseudo-label)
                pseudo_utterance = f"1 {token}"
                self.answer(scene, pos, pseudo_utterance)
                
                if verbose:
                    status = "KNOWN" if result.is_known else "NEW"
                    print(f"[SelfTrain] Scene {scene_idx}, Pos {pos}: "
                          f"{status} -> '{token}'")
        
        if verbose:
            print(f"\n[SelfTrain] 完成: known={stats['known']}, "
                  f"new={stats['new']}, empty={stats['empty']}")
        
        return stats
    
    # =========================================================================
    # Reflection (概念合并检测)
    # =========================================================================
    
    def _is_provisional_concept(self, token: str) -> bool:
        """判断是否是临时概念 (concept_xxxxx)。"""
        return token.startswith("concept_")
    
    def _symmetric_kl(self, concept_a, concept_b) -> float:
        """
        计算两个概念之间的对称 KL 散度。
        
        使用 KL(A||B) + KL(B||A) 的平均值作为对称度量。
        """
        from gaussian import kl_diag_gaussians
        
        kl_ab = kl_diag_gaussians(concept_a.mu, concept_a.var, 
                                   concept_b.mu, concept_b.var)
        kl_ba = kl_diag_gaussians(concept_b.mu, concept_b.var, 
                                   concept_a.mu, concept_a.var)
        return (kl_ab + kl_ba) / 2.0
    
    def reflect(self, z_threshold: float = 1.0, n_baseline_samples: int = 100, 
                allow_merge_trained: bool = False,
                verbose: bool = None) -> Dict[str, any]:
        """
        反思概念表：检测相似概念并合并。
        
        基于 KL 散度的 Z-score 判断概念相似性：
        - Z-score = (kl_baseline_mean - kl_pair) / kl_baseline_std
        - 如果 Z > threshold，说明两个概念非常相似，应该合并
        
        合并规则：
        1. 一个临时概念 + 一个训练概念 → 用训练概念的名字
        2. 两个都是训练概念 → 
           - allow_merge_trained=False: 不合并
           - allow_merge_trained=True: 合并，名字变成 frozenset({'red', 'blue'})
        3. 两个都是临时概念 → 随机选一个
        
        Args:
            z_threshold: Z-score 阈值 (默认 1.0)
            n_baseline_samples: 基准线采样数量
            allow_merge_trained: 是否允许合并两个训练概念 (默认 False)
            verbose: 是否打印详细信息
            
        Returns:
            统计信息: {
                "pairs_checked": int,
                "merges": List[Tuple[str, str, str]],  # (token_a, token_b, kept_token)
                "skipped_both_trained": int
            }
        """
        if verbose is None:
            verbose = self.verbose
            
        stats = {
            "pairs_checked": 0,
            "merges": [],
            "skipped_both_trained": 0
        }
        
        concepts = list(self.table._concepts.items())
        n_concepts = len(concepts)
        
        if n_concepts < 2:
            if verbose:
                print("[Reflect] 概念数量不足，无需检测。")
            return stats
        
        # 1. 建立基准线：采样随机概念对的 KL 散度
        from gaussian import kl_diag_gaussians
        
        baseline_kls = []
        for _ in range(n_baseline_samples):
            # 采样两个随机概念
            idx_a, idx_b = np.random.choice(n_concepts, size=2, replace=False)
            concept_a = concepts[idx_a][1]
            concept_b = concepts[idx_b][1]
            kl = self._symmetric_kl(concept_a, concept_b)
            baseline_kls.append(kl)
        
        mu_baseline = np.mean(baseline_kls)
        sigma_baseline = np.std(baseline_kls)
        
        # 特殊情况: 只有2个概念时，sigma=0，无法计算有效的 Z-score
        # 此时使用绝对 KL 阈值 (KL < 2.0 视为相似)
        use_absolute_threshold = sigma_baseline < 1e-6
        absolute_kl_threshold = 2.0  # 可调整
        
        if verbose:
            if use_absolute_threshold:
                print(f"[Reflect] 只有 {n_concepts} 个概念，使用绝对 KL 阈值 < {absolute_kl_threshold}")
            else:
                print(f"[Reflect] 基准线 KL: μ={mu_baseline:.2f}, σ={sigma_baseline:.2f}")
        
        # 2. 遍历所有概念对，检测相似性
        merged_tokens = set()  # 已被合并的 token
        
        for i in range(n_concepts):
            token_a, concept_a = concepts[i]
            
            if token_a in merged_tokens:
                continue
                
            for j in range(i + 1, n_concepts):
                token_b, concept_b = concepts[j]
                
                if token_b in merged_tokens:
                    continue
                
                stats["pairs_checked"] += 1
                
                # 计算 KL 散度
                kl = self._symmetric_kl(concept_a, concept_b)
                
                # 判断是否相似
                if use_absolute_threshold:
                    # 使用绝对阈值
                    is_similar = kl < absolute_kl_threshold
                    z_score = 0.0  # 用于显示
                else:
                    # 计算 Z-score (注意：KL 越小越相似，所以用 baseline - kl)
                    z_score = (mu_baseline - kl) / (sigma_baseline + 1e-6)
                    is_similar = z_score > z_threshold
                
                if is_similar:
                    # 相似！检查合并规则
                    is_a_provisional = self._is_provisional_concept(token_a)
                    is_b_provisional = self._is_provisional_concept(token_b)
                    
                    if not is_a_provisional and not is_b_provisional:
                        # 两个都是训练概念
                        if not allow_merge_trained:
                            stats["skipped_both_trained"] += 1
                            if verbose:
                                print(f"[Reflect] 跳过: '{token_a}' 和 '{token_b}' 都是训练概念 (Z={z_score:.2f})")
                            continue
                        else:
                            # 允许合并训练概念，使用 frozenset 作为新名字
                            # 解析现有名字中的 token 集合
                            tokens_a = self._parse_merged_token(token_a)
                            tokens_b = self._parse_merged_token(token_b)
                            merged_name = frozenset(tokens_a | tokens_b)
                            
                            # 执行合并到新名字
                            self._merge_concepts_to_new(token_a, token_b, merged_name)
                            merged_tokens.add(token_a)
                            merged_tokens.add(token_b)
                            
                            stats["merges"].append((token_a, token_b, merged_name))
                            
                            if verbose:
                                print(f"[Reflect] 合并训练概念: '{token_a}' + '{token_b}' → {merged_name} (Z={z_score:.2f})")
                            continue
                    
                    # 安全检查：确保两个 token 都还存在于概念表中
                    if token_a not in self.table._concepts or token_b not in self.table._concepts:
                        if verbose:
                            print(f"[Reflect] 跳过: '{token_a}' 或 '{token_b}' 已被合并")
                        continue
                    
                    # 决定保留哪个
                    if is_a_provisional and not is_b_provisional:
                        # 规则 1: B 是训练概念，用 B
                        keep_token, remove_token = token_b, token_a
                    elif not is_a_provisional and is_b_provisional:
                        # 规则 1: A 是训练概念，用 A
                        keep_token, remove_token = token_a, token_b
                    else:
                        # 规则 3: 两个都是临时概念，保留 kappa 更大的
                        kappa_a = self.table._concepts[token_a].kappa
                        kappa_b = self.table._concepts[token_b].kappa
                        if kappa_a > kappa_b:
                            keep_token, remove_token = token_a, token_b
                        elif kappa_b > kappa_a:
                            keep_token, remove_token = token_b, token_a
                        else:
                            # kappa 相等时随机选
                            if np.random.random() < 0.5:
                                keep_token, remove_token = token_a, token_b
                            else:
                                keep_token, remove_token = token_b, token_a
                    
                    # 执行合并：将 remove 的统计量合并到 keep 中
                    self._merge_concepts(keep_token, remove_token)
                    merged_tokens.add(remove_token)
                    
                    stats["merges"].append((token_a, token_b, keep_token))
                    
                    if verbose:
                        print(f"[Reflect] 合并: '{remove_token}' → '{keep_token}' (Z={z_score:.2f}, KL={kl:.2f})")
        
        # 清除背景统计缓存
        self._bg_stats = None
        
        if verbose:
            print(f"\n[Reflect] 完成: 检查 {stats['pairs_checked']} 对, "
                  f"合并 {len(stats['merges'])} 对, "
                  f"跳过 {stats['skipped_both_trained']} 对 (都是训练概念)")
        
        return stats
    
    def _merge_concepts(self, keep_token: str, remove_token: str) -> None:
        """
        将 remove_token 的统计量合并到 keep_token 中，然后删除 remove_token。
        
        使用加权平均合并均值，并更新方差。
        """
        keep_concept = self.table._concepts[keep_token]
        remove_concept = self.table._concepts[remove_token]
        
        # 加权合并
        kappa_k = keep_concept.kappa
        kappa_r = remove_concept.kappa
        kappa_new = kappa_k + kappa_r
        
        if kappa_new > 0:
            # 加权均值
            mu_new = (kappa_k * keep_concept.mu + kappa_r * remove_concept.mu) / kappa_new
            
            # 合并方差 (使用并行算法)
            delta = remove_concept.mu - keep_concept.mu
            var_new = (kappa_k * keep_concept.var + kappa_r * remove_concept.var) / kappa_new
            var_new += (kappa_k * kappa_r / kappa_new) * (delta ** 2)
            
            keep_concept.mu = mu_new
            keep_concept.var = var_new
            keep_concept.kappa = kappa_new
        
        # 删除被合并的概念
        del self.table._concepts[remove_token]
        
        # 更新 pending_concepts 中的引用
        for key, token in list(self.pending_concepts.items()):
            if token == remove_token:
                self.pending_concepts[key] = keep_token
    
    def _parse_merged_token(self, token) -> set:
        """
        解析 token 名称，提取其中包含的所有原始 token。
        
        如果 token 是 frozenset，返回其内容；
        如果是字符串，返回包含该字符串的 set。
        """
        if isinstance(token, frozenset):
            return set(token)
        else:
            return {token}
    
    def _merge_concepts_to_new(self, token_a, token_b, new_name) -> None:
        """
        将两个概念合并到一个新名字的概念中，然后删除原来的两个。
        
        Args:
            token_a: 第一个概念的 token
            token_b: 第二个概念的 token
            new_name: 新的概念名 (通常是 frozenset)
        """
        from concepts import Concept
        
        concept_a = self.table._concepts[token_a]
        concept_b = self.table._concepts[token_b]
        
        # 加权合并
        kappa_a = concept_a.kappa
        kappa_b = concept_b.kappa
        kappa_new = kappa_a + kappa_b
        
        if kappa_new > 0:
            # 加权均值
            mu_new = (kappa_a * concept_a.mu + kappa_b * concept_b.mu) / kappa_new
            
            # 合并方差 (使用并行算法)
            delta = concept_b.mu - concept_a.mu
            var_new = (kappa_a * concept_a.var + kappa_b * concept_b.var) / kappa_new
            var_new += (kappa_a * kappa_b / kappa_new) * (delta ** 2)
        else:
            mu_new = concept_a.mu
            var_new = concept_a.var
        
        # 创建新概念
        new_concept = Concept(
            token=str(new_name),  # frozenset 转为字符串用于显示
            mu=mu_new,
            var=var_new,
            kappa=kappa_new
        )
        
        # 添加新概念 (用 frozenset 作为 key)
        self.table._concepts[new_name] = new_concept
        
        # 删除旧概念
        del self.table._concepts[token_a]
        del self.table._concepts[token_b]
        
        # 更新 pending_concepts 中的引用
        for key, token in list(self.pending_concepts.items()):
            if token == token_a or token == token_b:
                self.pending_concepts[key] = new_name
    
    # =========================================================================
    # Query Methods (查询辅助)
    # =========================================================================
    
    def get_pending_concepts(self) -> Dict[Tuple[int, int], str]:
        """获取所有待确认的临时概念。"""
        return self.pending_concepts.copy()
    
    def clear_pending(self):
        """清空所有待确认的临时概念。"""
        self.pending_concepts.clear()
    
    def concept_stats(self) -> Dict[str, dict]:
        """获取所有概念的统计信息。"""
        stats = {}
        for token, concept in self.table._concepts.items():
            stats[token] = {
                "count": getattr(concept, 'count', 1),
                "mu_norm": np.linalg.norm(concept.mu),
                "var_mean": np.mean(concept.var)
            }
        return stats


# =============================================================================
# 便捷函数
# =============================================================================

def create_active_learner(table: ConceptTable, 
                          novelty_threshold: float = 3.0,
                          min_kappa_known: float = 2.0) -> ActiveLearner:
    """创建 ActiveLearner 实例。"""
    return ActiveLearner(table, novelty_threshold=novelty_threshold, 
                         min_kappa_known=min_kappa_known)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """演示 Active Learning 功能。"""
    import numpy as np
    from templates import SHAPES, COLORS_RGB
    from world import Obj, Scene
    
    def make_obj(shape, color):
        return Obj(shape_name=shape, 
                   color_rgb=COLORS_RGB[color], 
                   occ=np.array(SHAPES[shape], dtype=np.float32))
    
    print("=" * 60)
    print("Active Learning Demo")
    print("=" * 60)
    
    # 初始化
    table = ConceptTable(d=12)
    learner = ActiveLearner(table, novelty_threshold=0.5)
    
    # 训练一些已知概念
    print("\n--- 训练已知概念 ---")
    for s in ['box', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(s, 'red'), None, None, None]))
        learn_step(X, m, k=1, tokens=['red'], table=table)
        print(f"训练: red {s}")
    
    for s in ['box', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(s, 'blue'), None, None, None]))
        learn_step(X, m, k=1, tokens=['blue'], table=table)
        print(f"训练: blue {s}")
    
    # 场景: 红色方块(已知) + 粉色方块(未知)
    print("\n--- Ask 测试 ---")
    scene = Scene(regions=[make_obj('box', 'red'), make_obj('box', 'pink'), None, None])
    
    # Ask 位置 0 (红色 - 已知)
    result0 = learner.ask(scene, 0)
    print(f"\nPosition 0 (red box):")
    print(f"  {result0.message}")
    
    # Ask 位置 1 (粉色 - 未知)
    result1 = learner.ask(scene, 1)
    print(f"\nPosition 1 (pink box):")
    print(f"  {result1.message}")
    
    # Answer 测试
    print("\n--- Answer 测试 ---")
    msg = learner.answer(scene, 1, "1 pink")
    print(f"Answer '1 pink' for position 1: {msg}")
    
    print("\n--- 概念统计 ---")
    for token, stats in learner.concept_stats().items():
        print(f"  {token}: count={stats['count']}")
    
    print("\nDemo 完成!")


if __name__ == "__main__":
    demo()
