"""渐进式 IE-RSA: 每轮递减 max_concepts"""
import sys
import numpy as np
sys.path.insert(0, 'MNIST')

from mnist_encoder import encode_mnist_batch, load_mnist_data
from ie_rsa import IERSAConceptTable, NoveltyDetector, EpochStats, evaluate_clustering

np.random.seed(42)
e = np.e

# 加载数据
X_raw, y = load_mnist_data('MNIST/.mnist_cache', n_samples=5000)
X = encode_mnist_batch(X_raw)
print(f'Data: {X.shape}')

# 渐进式 max_concepts 设置
max_concepts_schedule = [1000, 500, 250, 120, 60, 30, 15, 10]

print('\n' + '='*50)
print('Progressive IE-RSA: Decreasing max_concepts')
print('='*50)
print(f'Schedule: {max_concepts_schedule}')

n_samples, d = X.shape
z_threshold = 0.3
seed_kappa = 5.0

survivors = []
k_prev = 0
epoch_stats = []

for epoch, max_concepts in enumerate(max_concepts_schedule, 1):
    print(f'\n=== Epoch {epoch} (max={max_concepts}) ===')
    
    # Step A: 创建表并继承
    table = IERSAConceptTable(d=d)
    for i, (token, concept) in enumerate(survivors):
        table.add_concept_manual(f"seed_{epoch}_{i}", concept.mu, concept.var, kappa=seed_kappa)
    
    if survivors:
        print(f'Inherited {len(survivors)} concepts')
    
    # Step B: 校准检测器
    detector = NoveltyDetector(d=d, z_threshold=z_threshold)
    n_calib = min(100, n_samples)
    calib_idx = np.random.choice(n_samples, n_calib, replace=False)
    detector.calibrate(X[calib_idx], table)
    
    # Step C: 流式训练
    perm = np.random.permutation(n_samples)
    n_known = 0
    n_new = 0
    
    for i, idx in enumerate(perm):
        x = X[idx]
        
        is_known, best_token, z_score = detector.check(x, table)
        
        if is_known and best_token is not None:
            table.update_concept(best_token, x)
            n_known += 1
        else:
            if len(table) >= max_concepts:
                n_new += 1
                print(f'  Concept limit ({max_concepts}), starting pruning...')
                break
            new_token = f"c{epoch}_{table.next_id()}"
            table.add_concept(new_token, x)
            n_new += 1
        
        # 重新校准
        if (i + 1) % 500 == 0:
            detector.calibrate(X[calib_idx], table)
    
    N_t = len(table)
    print(f'Training: {n_known} known, {n_new} new, total={N_t}')
    
    # Step D: 剪枝目标 (强制不超过当前 max_concepts)
    if epoch == 1:
        k_target = int(N_t / e)
    else:
        k_target = int((k_prev + N_t) / 2)
    # 关键：强制压缩到 max_concepts
    k_target = max(2, min(k_target, N_t, max_concepts))
    
    # Step E: 剪枝
    concepts = list(table._concepts.items())
    concepts.sort(key=lambda x: x[1].kappa, reverse=True)
    survivors = concepts[:k_target]
    
    print(f'Kept {len(survivors)} concepts')
    if survivors:
        top5 = survivors[:5]
        for j, (token, c) in enumerate(top5):
            print(f'  {j+1}. {token}: kappa={c.kappa:.1f}')
    
    stats = EpochStats(epoch, N_t, len(survivors), n_known, n_new)
    epoch_stats.append(stats)
    k_prev = len(survivors)

# 重建最终表
final_table = IERSAConceptTable(d=d)
for token, concept in survivors:
    final_table.add_concept_manual(f"final_{final_table.next_id()}", 
                                    concept.mu, concept.var, concept.kappa)

# 评估
print('\n' + '='*50)
print('EVALUATION')
print('='*50)

metrics = evaluate_clustering(final_table, X, y)
print(f"Final Concepts: {metrics['n_concepts']}")
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Epoch 历史
print("\nEpoch History:")
print(f"{'Epoch':>6} {'MaxC':>6} {'Generated':>10} {'Kept':>6} {'Known':>6} {'New':>6}")
print('-'*50)
for i, s in enumerate(epoch_stats):
    print(f"{s.epoch:>6} {max_concepts_schedule[i]:>6} {s.n_generated:>10} {s.n_kept:>6} {s.n_known:>6} {s.n_new:>6}")
