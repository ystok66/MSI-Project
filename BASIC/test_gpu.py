"""GPU 版本快速测试"""
import sys
import time
import numpy as np
sys.path.insert(0, 'MNIST')

from mnist_encoder import encode_mnist_batch, load_mnist_data
from ie_rsa_gpu import iterative_evolutionary_clustering_gpu, evaluate_clustering_gpu

np.random.seed(42)

print("Loading data...")
X_raw, y = load_mnist_data('MNIST/.mnist_cache', n_samples=5000)
X = encode_mnist_batch(X_raw)
print(f"Data: {X.shape}")

print("\nRunning GPU version...")
t0 = time.time()
table, stats = iterative_evolutionary_clustering_gpu(
    X, 
    max_epochs=10,
    z_threshold=0.3,
    max_concepts_per_epoch=2000,
    verbose=True
)
t1 = time.time()

metrics = evaluate_clustering_gpu(table, X, y)

print(f"\n{'='*50}")
print(f"GPU Time: {t1-t0:.2f}s")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Concepts: {metrics['n_concepts']}")
