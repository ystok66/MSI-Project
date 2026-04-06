"""
config.py — Hyperparameters and defaults for the CLS system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CLSConfig:
    """All tunable knobs for CLSAgent, grouped by layer."""

    # ── General ──
    mode: str = 'ast'              # 'stack' or 'ast'
    use_hpc: bool = True           # enable Layer 2
    n_em: int = 3                  # EM iterations

    # ── Layer 1: Cortex ──
    beam_k: int = 10               # top-K traces to keep (match NSLearner)
    beam_width: int = 30           # beam width per step (match NSLearner)
    decay_rate: float = 0.5        # role/repeat count decay between EM iters
    rsa_alpha: float = 0.0         # RSA rationality (0 = no RSA for speed)

    # ── Emission model ──
    gauss: bool = False            # True = CIELAB Gaussian emission (3D Lab vec)
    delta: Optional[Dict[str, float]] = None  # Discrete Dirichlet prior (None = NIG/KL)
    lab_sigma: float = 0.0         # Gaussian noise σ in raw Lab units (0 = no noise)

    # Dirichlet priors
    alpha: Dict[str, float] = field(default_factory=lambda: {
        'EMIT': 2.0,
        'REPEAT': 1.0,
        'SWAP_INFIX': 0.3,
        'CONCAT_INFIX': 0.3,
        'OVER_INFIX': 0.3,
    })

    # ── Layer 2: HPC ──
    hpc_d_bow: int = 64            # BOW hash dimensions
    hpc_d_bigr: int = 64           # bigram hash dimensions
    hpc_m: int = 512               # DG sparse code dimension
    hpc_k: int = 30                # kWTA sparsity
    hpc_noise_std: float = 0.01    # DG noise
    hpc_top_r: int = 5             # CA3 retrieval top-R
    hpc_eta: float = 1.0           # Hopfield learning rate
    hpc_completion_steps: int = 3  # CA3 pattern completion iterations
    hpc_lam_min: float = 0.0      # CA1 gate minimum λ
    hpc_lam_max: float = 1.0      # CA1 gate maximum λ
    hpc_seed: int = 42

    # CA1 Mahalanobis (blockwise + ridge)
    ca1_eps: float = 1e-3          # ridge regularization for inv_var
    ca1_mix_a: float = 0.7         # residual vs feature var mixing weight
    ca1_default_th: float = 0.5
    ca1_default_temp: float = 0.1

    # Replay
    replay_batch_size: int = 3
    replay_lr: float = 0.2         # weight for replay soft updates
    replay_priority_rho: float = 0.3  # mixing: (1-ρ)*uniform + ρ*priority
    replay_priority_clip: float = 5.0  # clip priority scores

    # ── E-step IS correction & normalization ──
    use_is_correction: bool = False    # IS correction: log_w = log_p - log_q (opt-in)
    norm_by_steps: bool = False        # per-step averaging of score components (opt-in)
    T_resp_base: float = 1.0           # base temperature for responsibilities softmax
    T_resp_min: float = 0.5            # min temperature clamp
    T_resp_max: float = 2.0            # max temperature clamp
    T_resp_scale_by_support: bool = False  # scale T by sqrt(1 + n_support/10) (opt-in)

    # ── Cerebellum episode-level tuning ──
    cerebellum_lr: float = 0.05        # episode-level weight adjustment lr
    cerebellum_enable: bool = False    # disabled by default in v0

    # ── Layer 3: Control ──
    # BG selector
    bg_explore_factor: float = 1.0     # beam expansion multiplier in explore mode
    bg_max_beam_expand: int = 24       # cap for expanded beam
    bg_rsa_rerank_alpha: float = 0.0   # RSA utility rerank weight (0 = off)
    bg_rsa_cost_per_op: float = 0.01   # per-operation cost for RSA utility
