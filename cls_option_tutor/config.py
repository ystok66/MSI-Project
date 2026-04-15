"""
config.py — All hyperparameters for the CLS Option Tutor V2.

V2 changes:
  - HP_0 = 5 (from 10)
  - max_refreshes = 2 (from 3)
  - Discrete risk classes: 6 safe + 4 risky
  - Budget-aware learner (alpha_ko, alpha_time)
  - RISK_HINT replaces BAN (c_ban archived, c_hint added)
  - Hazard head + severity head replaces single danger head
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Core environment parameters."""
    # Menu
    K: int = 10                        # candidates per menu
    T_max: int = 5                     # max rounds per query
    H_0: int = 5                       # initial HP per query (V2: was 10)

    # Block structure
    N_sup: int = 14                    # support examples per block
    M_queries: int = 8                 # queries per block
    N_obs: int = 2                     # Phase 2: observation (tutor watches frozen learner)
    N_teach: int = 3                   # Phase 3: teaching (tutor intervenes, learner learns)
    N_eval: int = 3                    # Phase 4: evaluation (frozen learner, no tutor)

    # Risk model (V2 discrete)
    n_safe: int = 6                    # safe options per menu
    n_risky: int = 4                   # risky options per menu
    risk_classes: tuple = (1, 2, 3, 4) # possible risk values for risky options
    danger_dim: int = 16               # dimension of danger vector v
    cluster_sigma: float = 0.5         # intra-cluster noise for danger vectors

    # Refresh (V2: risk-only)
    max_refreshes: int = 2             # max refreshes per query (V2: was 3)

    # Budget-based teach phase (0 = disabled, use fixed N_teach instead)
    teach_step_budget: int = 0         # total pick+refresh steps allowed in teach phase

    # ── Feedback mode ────────────────────────────────────────────────────
    # Controls whether wrong-pick reveals expose true output to learner.
    # "reveal"    : wrong picks trigger RevealEvent with true output (current default)
    # "nonreveal" : wrong picks do NOT expose true output; learner only knows "wrong"
    # Default = "reveal" → fully backward-compatible with all existing experiments.
    feedback_mode: str = "reveal"      # "reveal" | "nonreveal"


@dataclass
class LearnerConfig:
    """Learner policy parameters."""
    # Semantic weights
    alpha_sem: float = 1.0             # semantic score weight
    alpha_risk: float = 0.5            # danger prediction weight
    alpha_unc: float = 0.2             # danger uncertainty weight
    alpha_nov: float = 0.0             # novelty bonus (off in v2)
    alpha_ko: float = 1.0             # KO risk weight (V2 new)
    alpha_time: float = 0.3            # time pressure penalty (V2 new)
    beta_L: float = 4.0               # softmax temperature
    epsilon: float = 0.05              # lapse rate
    c_refresh: float = 0.3            # refresh cost
    tau_sem: float = 1.0               # semantic mismatch temperature

    # Root-cause disentangling modes
    reveal_learning_mode: str = "cortex_em"  # "cortex_em" | "off" | "negative_memory" | "nonreveal_negative"
    attention_init_mode: str = "uniform"     # "uniform" | "persistent_prior"
    eta_attn: float = 0.3                    # persistent prior strength
    alpha_neg: float = 2.0                   # negative memory penalty weight
    eta_reveal: float = 1.0                  # reveal update gate in [0,1]:
                                             # probability that a wrong-pick reveal
                                             # triggers incremental_study().
                                             # 1.0 = always (default); 0.0 = never.

    # ── nonreveal negative evidence ──────────────────────────────────────
    # Active when feedback_mode="nonreveal" and reveal_learning_mode="nonreveal_negative".
    # Learner cannot see revealed_output; instead records (program, target_output) pairs
    # as negative evidence and penalises them in future scoring.
    #
    # negative_evidence_mode:
    #   "off"                   : nonreveal + no learning (pure behaviour control)
    #   "exact_program_target"  : (program_tuple, target_output_tuple) keyed penalty
    negative_evidence_mode: str = "off"       # "off" | "exact_program_target"
    eta_negative: float = 1.0                 # weight per negative-evidence addition
                                              # (if None at runtime → fall back to eta_reveal)
    lambda_neg: float = 1.0                   # penalty scale applied in score_option()

    # ── Correct-pick learning (positive reinforcement) ───────────────────
    # Controls whether CLS incremental_study is called when learner picks correctly.
    # Provides (j*.text, target_output) as a positive supervision signal.
    # "off"       : no update on correct pick (default, backward-compat)
    # "cortex_em" : same EM pathway as wrong-pick reveal
    # NOTE: should be paired with nonreveal feedback_mode for best isolation.
    # In reveal mode, this may also strengthen no_tutor baseline (see user guide).
    correct_pick_learning_mode: str = "off"    # "off" | "cortex_em"
    eta_correct_pick: float = 1.0             # stochastic gate: P(CLS update | correct pick)
    correct_pick_n_em_override: int = 1       # lighter EM than wrong-reveal (default n_em)

    # ── Step 4: Persistent Highlight Prior ───────────────────────────────
    # Cross-query EMA attention bias from HIGHLIGHT events.
    # m_{t+1} = (1-ρ_hl)*m_t + ρ_hl * φ_hl(H_t)
    # Applied at query init: attention prior w_ℓ ∝ 1 + λ_hl * m_t[ℓ]
    # 0.0 = off (default, backward compatible)
    rho_hl_prior: float = 0.0       # EMA update rate  ρ_hl  ∈ [0,1]
    lambda_hl_prior: float = 0.3    # prior injection weight  λ_hl ≥ 0

    # ── Step 5: Persistent Ban Prior ─────────────────────────────────────
    # Cross-query EMA negative bias from BAN events.
    # n_{t+1} = (1-ρ_ban)*n_t + ρ_ban * φ_ban(banned_option_t)
    # Applied at scoring: U'_j -= λ_ban * dot(n_t, g_j)
    # 0.0 = off (default, backward compatible)
    rho_ban_prior: float = 0.0      # EMA update rate  ρ_ban ∈ [0,1]
    lambda_ban_prior: float = 0.5   # ban penalty weight  λ_ban ≥ 0

    # CLS semantic subsystem
    use_cls: bool = False              # False = deterministic baseline
    n_sup: int = 5                     # support examples to show (1/3/5)
    n_em: int = 2                      # EM iterations for CLS
    use_hpc: bool = True               # HPC memory for CLS
    rho_H: float = 2.0                  # highlight attention boost (mirrored from tutor)

    # Hazard head (V2: binary safe/risky classifier)
    hazard_lr: float = 0.1             # learning rate for hazard head
    hazard_prior_var: float = 1.0      # prior variance
    eta_hint: float = 0.8             # RISK_HINT weak label confidence

    # Severity head (V2: damage regression given risky)
    severity_prior_var: float = 1.0
    severity_lr: float = 0.1


@dataclass
class TutorConfig:
    """Tutor policy parameters."""
    # Scoring weights
    beta_corr: float = 2.0             # correctness improvement weight
    beta_hp: float = 1.0               # HP safety weight
    beta_learn: float = 0.5            # learning gain weight
    beta_time: float = 0.3             # time pressure weight
    beta_safe: float = 1.5             # safety gain weight
    beta_IG: float = 1.0               # HIGHLIGHT info gain weight
    beta_over: float = 0.2             # HIGHLIGHT over-reveal penalty

    # Intervention fixed costs
    c_ban: float = 0.0                 # BAN cost (equal to HIGHLIGHT)
    c_hint: float = 0.3                # RISK_HINT cost (V2 new)
    c_hl: float = 0.0                  # V2: free (cost via beta_over only)
    c_skip: float = 1.5

    # SKIP mastery parameters
    beta_mastery: float = 0.8
    beta_certainty: float = 0.4

    # HIGHLIGHT
    max_highlight_cells: int = 2
    rho_H: float = 2.0                 # highlight attention boost

    # Profile inference
    profile_grid_size: int = 5         # grid points per profile dim

    # Root-cause disentangling modes
    tutor_access_mode: str = "proxy_oracle"  # "proxy_oracle" | "cheat_sem" | "cheat_full"

    # Eval-aware Bayesian tutor (P0 / P1 / P2)
    tutor_scorer_mode: str = "legacy"  # "legacy" | "eval_aware"
    lambda_now: float = 1.0            # weight for current-query utility (R_now)
    lambda_probe: float = 0.0         # weight for probe delta (ΔProbe); 0 = legacy
    n_probe: int = 30                  # held-out probe queries for eval surrogate (P1: 30-50)
    probe_seed: int = 99               # fixed seed for probe generation
    probe_ood_ratio: float = 0.5       # fraction of probes that are OOD (P1)
    probe_use_accuracy: bool = True    # True = eval_score() (P1 acc-based); False = score() (legacy margin)
    # P2: short-horizon rollout
    shadow_rollout_horizon: int = 1    # H=1: single-step (P1); H=2: two-step lookahead (P2)
    shadow_rollout_gamma: float = 1.0  # discount factor for step-t+1 probe gain (1.0 = no discount)

    # ── Bayes Gate Tutor (Stage A) ───────────────────────────────
    # Q_use = λ_eval*G_eval + λ_exp*G_exp - β*P_death - γ*P_timeout - λ_shift*D_shift - c(a)
    lambda_eval: float = 1.0           # weight for ΔEval gain (ProbeEval or OracleSurrogate)
    lambda_exp: float = 0.25           # weight for safe exposure gain (G_exp)
    lambda_shift: float = 0.125        # weight for distribution shift penalty (D_shift JS) — lowered from 0.25
    c_I: float = 0.05                  # unified intervention cost per action slot (BAN, HL each count 1)

    # ρ_H for HIGHLIGHT: tutor-side planning model of highlight effect size.
    # None → falls back to LearnerConfig.rho_H at runtime.
    tutor_rho_H: float = 2.0           # used in logit surrogate; set to LearnerConfig.rho_H default

    M_ban: float = 50.0               # BAN logit suppression in planning surrogate (≈ remove)

    # Sparse tutor candidate generation
    n_sparse_candidates: int = 4       # WAIT / BAN / HL / MIX
    hl_timeout_threshold: float = 0.5  # generate HIGHLIGHT candidate only if P_timeout(WAIT)>this

    # G_learn mode for sparse tutor (same enum as OptionLevelTutorAgent)
    sparse_g_learn_mode: str = "none"  # "none" | "probe" | "oracle_surrogate"
    sparse_n_probe: int = 20           # probe queries for mode="probe"

    # ── Rollout-based survival proxy ─────────────────────────────────────
    # Replaces geometric P_timeout and single-step P_death with short rollouts
    # that use the learner's actual scoring model (scorer + danger_head + attention).
    # rollout_mode:
    #   "proxy"  : old static approximation (backward-compatible baseline)
    #   "hybrid" : rollout for rescue mode + decision-boundary cases; proxy elsewhere
    #   "full"   : rollout for all non-WAIT candidates
    rollout_mode: str = "hybrid"       # "proxy" | "hybrid" | "full"
    rollout_n: int = 8                 # rollouts per candidate (smoke: 4, full: 8)
    rollout_delta: float = 0.05        # margin threshold: rollout when |Q_nonwait - Q_wait| < delta

    # ── G_learn feedback awareness ────────────────────────────────────────
    # Controls which feedback assumption the tutor's G_learn estimator uses.
    # "inherit" : follow cfg.env.feedback_mode (default, always consistent)
    # "reveal"  : force reveal-aware estimation regardless of env setting
    # "nonreveal": force nonreveal-aware estimation regardless of env setting
    g_learn_feedback_mode: str = "inherit"  # "inherit" | "reveal" | "nonreveal"

    # ── Dual-mode tutor objective ─────────────────────────────────────────
    # tutor_mode:
    #   "current"     : existing unified Q_use scalar (default, backward-compat)
    #   "protective"  : U_teach maximized subject to eval non-regression guard
    #   "pedagogical" : U_learn + η*U_teach subject to hard safety constraints
    tutor_mode: str = "current"

    # U_teach weights (shared by protective and pedagogical).
    # U_teach = w_succ*p_success - w_death_teach*p_death - w_tout_teach*p_timeout
    # p_success, p_death, p_timeout all come from learner-consistent rollout.
    w_succ: float = 1.0             # p_success contribution
    w_death_teach: float = 0.5      # P_death penalty in U_teach
    w_tout_teach: float = 0.2       # P_timeout penalty in U_teach

    # Protective mode: eval non-regression guard.
    # Any non-WAIT action with G_eval(a) < -eps_eval_guard is filtered BEFORE argmax.
    # Guard is bypassed if g_learn_mode="none" (no eval signal available).
    eps_eval_guard: float = 0.01

    # Pedagogical mode: hard safety constraints.
    # Dynamic thresholds: d_max = min(p_death_wait + d_max_margin, d_max_cap)
    #                     t_max = min(p_timeout_wait + t_max_margin, t_max_cap)
    # Actions with P_death > d_max OR P_timeout > t_max are filtered BEFORE argmax.
    eta_pedagogical: float = 0.25   # U_learn weight in mixed objective
    d_max_margin: float = 0.01      # allowed P_death increase vs WAIT baseline
    t_max_margin: float = 0.03      # allowed P_timeout increase vs WAIT baseline
    d_max_cap: float = 0.20         # absolute P_death cap (safety)
    t_max_cap: float = 0.70         # absolute P_timeout cap (safety)

    # ── Bayes Gate Tutor (Stage B) ───────────────────────────────
    max_passes_per_block: int = 2      # cap on PASS_QUERY actions per block
    pass_n_candidates: int = 4         # M: query candidates to sample for Q_pass estimate
    c_pass: float = 0.05               # PASS_QUERY cost (same scale as c_I)




@dataclass
class RSAConfig:
    """RSA L1 learner + L0 tutor hyperparameters.

    RSA pathway is OFF by default; set use_rsa=True to activate.
    All legacy modules remain functional when use_rsa=False.
    """
    # ── Master switch ────────────────────────────────────────────
    use_rsa: bool = False           # False = legacy path; True = L1 RSA listener

    # ── HIGHLIGHT → semantic RSA ─────────────────────────────────
    # P_S0(HL(H) | j) ∝ exp(omega_hl * s_HL(j;H))
    # s_HL(j;H) = -M_H(j) + lambda_ctx * M_barH(j)
    omega_hl: float = 2.0           # HIGHLIGHT log-likelihood strength
    lambda_ctx: float = 0.5         # contrastive weight for non-highlighted cells

    # ── Semantic gate (U-shape entropy gate) ─────────────────────
    # Gate type: "entropy" (canonical) | "none" (no gate)
    # U-shape gate: h = H(q_t^0) / log(K),  g = 4*h*(1-h)
    # Applied as: b_sem_gated(j) = g * b_sem(j)
    # Input: q_t^0 = base decision posterior (pre-RSA: CLS + risk + unc)
    #        NOT raw sem_scores — must include risk/unc channels
    use_sem_gate: bool = True
    sem_gate_type: str = "entropy"  # "entropy" | "none"
    # Research fallback: hard threshold gate (use only for ablation)
    # sem_gate_type = "threshold" uses gate_lo/hi
    sem_gate_lo: float = 0.25       # below: learner too confident (suppress bias)
    sem_gate_hi: float = 0.90       # above: learner too confused (suppress bias)

    # ── BAN → risk RSA ───────────────────────────────────────────
    # logit P(r_j=1 | BAN(j)) = logit P(r_j=1) + omega_ban
    omega_ban: float = 3.0          # BAN logit shift strength
    # ban_teaches_risk: parametric DangerHead update from BAN (cross-query)
    # DEFAULT: False — Exp F data shows no eval gain; B1=canonical-off, research-flag only
    ban_teaches_risk: bool = False

    # ── PASS ─────────────────────────────────────────────────────
    # PASS = tutor explicitly aborts query; learner receives pass_abort=True

    # ── meta-attention persistence ───────────────────────────────
    # bar_w_{t+1} = Normalize((1-rho)*bar_w_t + rho*u(H))
    # w_eff = Normalize((1-gamma)*w_query + gamma*bar_w)
    rho_attn: float = 0.3           # meta-attention update rate ρ
    gamma_attn: float = 0.3         # meta-prior blend ratio γ in query attention

    # ── L0 tutor ─────────────────────────────────────────────────
    use_l0_tutor: bool = False      # False = legacy tutor; True = L0 speaker
    lambda_task: float = 1.0        # λ_task: task utility weight in U_S0
    lambda_teach: float = 1.0       # λ_teach: teach utility weight in U_S0
    lambda_ko: float = 1.0          # λ_ko: KO penalty weight in G_task
    # ban_parametric_penalty: constant added to L0 BAN utility to discourage
    # BAN-heavy teaching (which produces high TEACH_SR but low EVAL transfer).
    # Set to 0.0 to disable. Exp G3c uses -0.3.
    ban_parametric_penalty: float = 0.0


@dataclass
class TrustState:
    """Future trust hook — reserved but disabled in V2."""
    enabled: bool = False
    tau: float = 0.3
    compliance_count: int = 0
    defiance_count: int = 0


@dataclass
class FullConfig:
    """Top-level config combining all subsystems."""
    env: EnvConfig = field(default_factory=EnvConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    tutor: TutorConfig = field(default_factory=TutorConfig)
    rsa: RSAConfig = field(default_factory=RSAConfig)

    # CLS backend
    cls_mode: str = "ast"
    cls_use_hpc: bool = True
    cls_gauss: bool = False
    cls_data_dir: str = ""

    # Reproducibility
    seed: int = 42
