"""双层 IE-RSA v2: 概念增强嵌入"""
import sys
import numpy as np
from scipy.special import softmax
sys.path.insert(0, 'MNIST')

from mnist_encoder import encode_mnist_batch, load_mnist_data
from ie_rsa import iterative_evolutionary_clustering, evaluate_clustering

np.random.seed(42)
e = np.e

# 加载数据
X_raw, y = load_mnist_data('MNIST/.mnist_cache', n_samples=5000)
X = encode_mnist_batch(X_raw)
print(f'Data: {X.shape}')

# ========== Layer 1: 原始数据聚类 ==========
print('\n' + '='*50)
print('LAYER 1: Raw data clustering')
print('='*50)

table_L1, stats_L1 = iterative_evolutionary_clustering(
    X, 
    max_epochs=5,
    z_threshold=0.3,
    max_concepts_per_epoch=1000,
    verbose=True
)

n_concepts_L1 = len(table_L1)
print(f'\nLayer 1 concepts: {n_concepts_L1}')

# 构建概念矩阵
concepts_L1 = list(table_L1._concepts.values())
C = np.array([c.mu for c in concepts_L1])  # (n_concepts, 196)

# ========== Layer 2 嵌入: 概念增强 ==========
print('\n' + '='*50)
print('Generating Layer 2 embeddings (concept-augmented)')
print('='*50)

def get_L2_embedding(x, C, top_k=10):
    """
    概念增强嵌入:
    embed = (1/(1+e)) * x + (e/(1+e)) * sum(softmax(sim_i) * mu_i)
    
    Args:
        x: 原始图片嵌入 (196,)
        C: 概念矩阵 (n_concepts, 196)
        top_k: 选择 top-k 个概念
    """
    # 计算相似度 (负 L2 距离)
    sims = -np.sum((C - x) ** 2, axis=1)  # (n_concepts,)
    
    # 选择 top-k
    top_indices = np.argsort(sims)[-top_k:]
    top_sims = sims[top_indices]
    top_concepts = C[top_indices]  # (top_k, 196)
    
    # Softmax 权重
    weights = softmax(top_sims)  # (top_k,)
    
    # 加权概念组合
    weighted_concept = np.sum(weights[:, None] * top_concepts, axis=0)  # (196,)
    
    # 混合: 1/(1+e) * x + e/(1+e) * weighted_concept
    alpha = 1 / (1 + e)
    beta = e / (1 + e)
    
    embed = alpha * x + beta * weighted_concept
    return embed

# 批量生成 L2 嵌入
X_L2 = np.array([get_L2_embedding(x, C, top_k=10) for x in X])
print(f'L2 embeddings: {X_L2.shape}')
print(f'Weights: img={1/(1+e):.4f}, concepts={e/(1+e):.4f}')

# ========== Layer 2: 概念增强空间聚类 ==========
print('\n' + '='*50)
print('LAYER 2: Concept-augmented clustering')
print('='*50)

table_L2, stats_L2 = iterative_evolutionary_clustering(
    X_L2,
    max_epochs=10,
    z_threshold=0.3,
    max_concepts_per_epoch=100,
    verbose=True
)

# ========== 评估 ==========
print('\n' + '='*50)
print('EVALUATION')
print('='*50)

metrics_L1 = evaluate_clustering(table_L1, X, y)
metrics_L2 = evaluate_clustering(table_L2, X_L2, y)

print(f"\nLayer 1: {metrics_L1['n_concepts']} concepts, ACC={metrics_L1['accuracy']:.2%}")
print(f"Layer 2: {metrics_L2['n_concepts']} concepts, ACC={metrics_L2['accuracy']:.2%}")

# 显示 epoch 详情
print("\nLayer 2 Epoch History:")
print(f"{'Epoch':>6} {'Generated':>10} {'Kept':>6} {'Known':>6} {'New':>6}")
print('-'*40)
for s in stats_L2:
    flag = ' *' if s.converged else ''
    print(f"{s.epoch:>6} {s.n_generated:>10} {s.n_kept:>6} {s.n_known:>6} {s.n_new:>6}{flag}")
