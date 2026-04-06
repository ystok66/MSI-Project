"""
Operator Table: 算子语义层 (Layer 2)

每个 token 存储一个线性变换的参数分布:
    y = A @ x + b

其中:
    x: 输入状态向量 (颜色序列的 one-hot 编码)
    y: 输出状态向量  
    A: 变换矩阵 — 学习置换/复制/删除等结构操作
    b: 偏置向量 — 学习颜色注入

参数用在线贝叶斯更新 (Welford) 维护均值+方差+计数:
    A_mu, A_var, b_mu, b_var, kappa

结构先验:
    A 初始化为 identity → 透传先验 (不变操作)
    行 softmax 正则化 → 鼓励近似置换矩阵
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class Operator:
    """单个算子: token → 线性变换 (A, b)."""
    token: str
    A_mu: np.ndarray       # (d_out, d_in) 变换矩阵均值
    b_mu: np.ndarray       # (d_out,) 偏置均值
    A_var: np.ndarray      # (d_out, d_in) 变换矩阵方差 (对角)
    b_var: np.ndarray      # (d_out,) 偏置方差
    kappa: float = 0.5     # 观测计数
    
    @property
    def confidence(self):
        """算子学习的置信度 (0~1)."""
        return min(1.0, self.kappa / 20.0)
    
    @property
    def is_identity_like(self):
        """是否接近恒等变换 (透传)."""
        d = min(self.A_mu.shape[0], self.A_mu.shape[1])
        I = np.eye(self.A_mu.shape[0], self.A_mu.shape[1])
        return np.linalg.norm(self.A_mu - I) < 0.5 and np.linalg.norm(self.b_mu) < 0.3


class OperatorTable:
    """
    算子词典.
    
    管理所有算子的参数, 支持:
    - ensure(token): 创建/获取算子
    - apply(token, x): 执行变换 y = A @ x + b
    - update(token, x, y, lr): 从 (x, y) 对梯度更新 A, b
    """
    
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self._operators: Dict[str, Operator] = {}
    
    def ensure(self, token: str) -> Operator:
        """获取算子, 不存在则创建 (identity 先验)."""
        token = token.lower()
        if token not in self._operators:
            # Identity 先验: A ≈ I, b ≈ 0
            A_mu = np.eye(self.d_out, self.d_in)
            b_mu = np.zeros(self.d_out)
            A_var = np.ones((self.d_out, self.d_in)) * 1.0  # 高不确定性
            b_var = np.ones(self.d_out) * 1.0
            self._operators[token] = Operator(
                token=token, A_mu=A_mu, b_mu=b_mu,
                A_var=A_var, b_var=b_var, kappa=0.5
            )
        return self._operators[token]
    
    def has(self, token: str) -> bool:
        return token.lower() in self._operators
    
    def get(self, token: str) -> Operator:
        return self._operators[token.lower()]
    
    def apply(self, token: str, x: np.ndarray) -> np.ndarray:
        """
        执行算子: y = A @ x + b
        
        Args:
            token: 算子名
            x: 输入状态 (d_in,)
            
        Returns:
            y: 输出状态 (d_out,)
        """
        op = self.ensure(token)
        y = op.A_mu @ x + op.b_mu
        return y
    
    def apply_sequence(self, token: str, seq: List[np.ndarray]) -> List[np.ndarray]:
        """
        对序列中的每个元素应用算子.
        
        Args:
            token: 算子名
            seq: 输入序列 [x1, x2, ...]
            
        Returns:
            输出序列 [y1, y2, ...]
        """
        return [self.apply(token, x) for x in seq]

    def update(self, token: str, x: np.ndarray, y: np.ndarray, 
               lr: float = 0.1) -> float:
        """
        从 (x, y) 对更新算子参数 (在线梯度下降).
        
        y_pred = A @ x + b
        loss = ||y - y_pred||^2
        dA = -(y - y_pred) @ x^T
        db = -(y - y_pred)
        
        Args:
            token: 算子名
            x: 输入 (d_in,)
            y: 期望输出 (d_out,)
            lr: 学习率
            
        Returns:
            MSE loss
        """
        op = self.ensure(token)
        
        # Forward
        y_pred = op.A_mu @ x + op.b_mu
        error = y - y_pred
        mse = np.mean(error ** 2)
        
        # Gradient update
        op.A_mu += lr * np.outer(error, x)
        op.b_mu += lr * error
        
        # Update kappa
        op.kappa += 0.1
        
        # 更新方差 (Welford-like for prediction variance)
        pred_var = error ** 2
        op.b_var += (pred_var - op.b_var) / op.kappa
        op.b_var = np.maximum(op.b_var, 1e-8)
        
        return mse

    def update_from_sequence(self, token: str, 
                             x_seq: List[np.ndarray],
                             y_seq: List[np.ndarray],
                             lr: float = 0.1) -> float:
        """
        从序列对 (x_seq → y_seq) 更新算子.
        
        关键: 如果 len(x) != len(y), 需要特殊处理:
        - len(y) > len(x): repeat 操作
        - len(y) < len(x): delete 操作
        - len(y) == len(x): 逐位置变换
        """
        total_mse = 0.0
        
        if len(x_seq) == len(y_seq):
            # 等长: 逐位置学
            for x, y in zip(x_seq, y_seq):
                total_mse += self.update(token, x, y, lr)
        elif len(y_seq) > len(x_seq) and len(x_seq) > 0:
            # 扩展: 学习重复模式 (用每个 x 预测对应的 y 段)
            ratio = len(y_seq) // len(x_seq)
            for i, x in enumerate(x_seq):
                for j in range(ratio):
                    idx = i * ratio + j
                    if idx < len(y_seq):
                        total_mse += self.update(token, x, y_seq[idx], lr)
        
        n = max(len(x_seq), len(y_seq), 1)
        return total_mse / n
    
    def tokens(self) -> List[str]:
        return list(self._operators.keys())
    
    def __len__(self):
        return len(self._operators)
