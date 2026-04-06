"""多层 IE-RSA + K-means 最终聚类"""
import sys
import numpy as np
from scipy.special import softmax
sys.path.insert(0, 'MNIST')

from mnist_encoder import encode_mnist_batch, load_mnist_data
from ie_rsa import iterative_evolutionary_clustering, IERSAConceptTable, evaluate_clustering

np.random.seed(42)
e = np.e

# 加载数据
X_raw, y = load_mnist_data('MNIST/.mnist_cache', n_samples=5000)
X = encode_mnist_batch(X_raw)
print(f'Raw data: {X.shape}')

def get_concept_augmented_embedding(X, concepts, top_k=10, alpha=1/(1+e), beta=e/(1+e)):
    """
    概念增强嵌入:
    embed = alpha * x + beta * sum(softmax(sim_i) * concept_i)
    """
    n_samples = X.shape[0]
    d = X.shape[1]
    C = np.array([c.mu for c in concepts])  # (n_concepts, d)
    
    embeddings = np.zeros((n_samples, d))
    
    for i, x in enumerate(X):
        # 计算相似度
        sims = -np.sum((C - x) ** 2, axis=1)
        
        # Top-k
        top_idx = np.argsort(sims)[-top_k:]
        top_sims = sims[top_idx]
        top_concepts = C[top_idx]
        
        # Softmax 权重
        weights = softmax(top_sims)
        
        # 加权概念
        weighted_concept = np.sum(weights[:, None] * top_concepts, axis=0)
        
        # 混合
        embeddings[i] = alpha * x + beta * weighted_concept
    
    return embeddings

# ========== Layer 1: 1000 concepts ==========
print('\n' + '='*60)
print('LAYER 1: 1000 concepts on raw data')
print('='*60)

table_L1, _ = iterative_evolutionary_clustering(
    X, max_epochs=5, z_threshold=0.3, max_concepts_per_epoch=1000, verbose=True
)
concepts_L1 = list(table_L1._concepts.values())
print(f'L1 concepts: {len(concepts_L1)}')

# ========== Layer 2: 300 concepts ==========
print('\n' + '='*60)
print('LAYER 2: 300 concepts (img + L1 top-10)')
print('='*60)

X_L2 = get_concept_augmented_embedding(X, concepts_L1, top_k=10)
print(f'L2 input shape: {X_L2.shape}')

table_L2, _ = iterative_evolutionary_clustering(
    X_L2, max_epochs=5, z_threshold=0.3, max_concepts_per_epoch=300, verbose=True
)
concepts_L2 = list(table_L2._concepts.values())
print(f'L2 concepts: {len(concepts_L2)}')

# ========== Layer 3: 100 concepts (L1 + L2) ==========
print('\n' + '='*60)
print('LAYER 3: 100 concepts (img + L1 + L2 top-10)')
print('='*60)

# L3 嵌入 = img + L1_aug + L2_aug
X_L2_aug = get_concept_augmented_embedding(X, concepts_L1, top_k=10)
X_L3_aug = get_concept_augmented_embedding(X_L2_aug, concepts_L2, top_k=10)
X_L3 = (1/(1+e)) * X + (e/(1+e)) * X_L3_aug
print(f'L3 input shape: {X_L3.shape}')

table_L3, _ = iterative_evolutionary_clustering(
    X_L3, max_epochs=5, z_threshold=0.3, max_concepts_per_epoch=100, verbose=True
)
concepts_L3 = list(table_L3._concepts.values())
print(f'L3 concepts: {len(concepts_L3)}')

# ========== Layer 4: 30 concepts ==========
print('\n' + '='*60)
print('LAYER 4: 30 concepts')
print('='*60)

X_L4_aug = get_concept_augmented_embedding(X_L3, concepts_L3, top_k=10)
X_L4 = (1/(1+e)) * X + (e/(1+e)) * X_L4_aug
print(f'L4 input shape: {X_L4.shape}')

table_L4, _ = iterative_evolutionary_clustering(
    X_L4, max_epochs=5, z_threshold=0.3, max_concepts_per_epoch=30, verbose=True
)
concepts_L4 = list(table_L4._concepts.values())
print(f'L4 concepts: {len(concepts_L4)}')

# ========== Final: K-means style 10 clusters ==========
print('\n' + '='*60)
print('FINAL: K-means style (K=10)')
print('='*60)

# 最终嵌入: img*(1/e) + L4*(e)
X_L4_final = get_concept_augmented_embedding(X_L4, concepts_L4, top_k=min(10, len(concepts_L4)))
X_final = (1/e) * X + (e/(1+e)) * X_L4_final
print(f'Final embedding: {X_final.shape}')

# K-means 风格: 固定 10 个中心
n_clusters = 10
d = X_final.shape[1]
n_samples = X_final.shape[0]

# 初始化: 随机选 10 个样本作为中心
init_idx = np.random.choice(n_samples, n_clusters, replace=False)
centers = X_final[init_idx].copy()

# 迭代更新
n_iters = 10
for it in range(n_iters):
    # 分配
    dists = np.sum((X_final[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, K)
    assignments = dists.argmin(axis=1)  # (N,)
    
    # 更新中心
    new_centers = np.zeros_like(centers)
    for k in range(n_clusters):
        mask = assignments == k
        if mask.sum() > 0:
            new_centers[k] = X_final[mask].mean(axis=0)
        else:
            new_centers[k] = centers[k]
    
    # 检查收敛
    diff = np.abs(new_centers - centers).max()
    centers = new_centers
    print(f'  Iter {it+1}: max center shift = {diff:.6f}')
    
    if diff < 1e-6:
        print(f'  Converged at iter {it+1}')
        break

# ========== 评估 ==========
print('\n' + '='*60)
print('EVALUATION')
print('='*60)

# 多数投票
from collections import Counter
cluster_labels = [[] for _ in range(n_clusters)]
for i, k in enumerate(assignments):
    cluster_labels[k].append(y[i])

mapping = []
for lbls in cluster_labels:
    if lbls:
        mapping.append(Counter(lbls).most_common(1)[0][0])
    else:
        mapping.append(-1)

correct = sum(mapping[assignments[i]] == y[i] for i in range(n_samples))
acc = correct / n_samples

print(f"\nFinal Clusters: {n_clusters}")
print(f"Accuracy: {acc:.2%}")

# 每个 cluster 的分布
print("\nCluster -> Digit Mapping:")
for k in range(n_clusters):
    count = sum(assignments == k)
    digit = mapping[k]
    print(f"  Cluster {k}: digit={digit}, count={count}")
