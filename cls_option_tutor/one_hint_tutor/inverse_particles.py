from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import copy

import numpy as np

from ..learner.cls_adapter import create_scorer
from ..learner.danger_head import DangerHead
from ..tutor.inverse_predictor import InverseShadowPredictor
from ..tutor.learner_model import ShadowLearnerModel
from ..tutor.profile_inference import posterior_probs
from .interfaces import ObservationRun, PosteriorSummary, TaskContext


@dataclass
class InversePosterior:
    predictor: InverseShadowPredictor
    summary: PosteriorSummary

    def profile_weights(self) -> List[Tuple[object, float]]:
        probs = posterior_probs(self.predictor._log_weights)
        return [
            (profile, float(probs[i]))
            for i, profile in enumerate(self.predictor._profiles)
        ]

    def planning_profile_weights(
        self,
        top_mass: float = 1.0,
        min_keep: int = 1,
    ) -> List[Tuple[object, float]]:
        weighted = sorted(
            self.profile_weights(),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        keep: List[Tuple[object, float]] = []
        mass = 0.0
        for profile, weight in weighted:
            keep.append((profile, float(weight)))
            mass += float(weight)
            if len(keep) >= max(1, int(min_keep)) and mass >= float(top_mass):
                break
        total = sum(weight for _, weight in keep) or 1.0
        return [(profile, weight / total) for profile, weight in keep]

    def profiles_for_stage(self, stage: str, cfg) -> List[Tuple[object, float]]:
        if stage == "refine":
            return self.planning_profile_weights(
                top_mass=float(getattr(cfg, "refine_profile_top_mass", 1.0)),
                min_keep=int(getattr(cfg, "refine_profile_min_keep", 1)),
            )
        return self.planning_profile_weights(
            top_mass=float(getattr(cfg, "profile_top_mass", 1.0)),
            min_keep=int(getattr(cfg, "profile_min_keep", 1)),
        )

    def cloned_shadow(self):
        shadow = self.predictor._shadow.deep_copy()
        scorer = getattr(shadow, "scorer", None)
        if scorer is not None and hasattr(scorer, "reset_debug_counters"):
            scorer.reset_debug_counters()
        return shadow


def _build_shadow_predictor(
    context: TaskContext,
    prelearn_examples,
    cfg,
) -> InverseShadowPredictor:
    scorer = create_scorer(
        context.grammar,
        list(prelearn_examples),
        use_cls=cfg.shadow_use_cls,
        n_sup=len(prelearn_examples),
        n_em=cfg.shadow_n_em,
        use_hpc=cfg.shadow_use_hpc,
        tau_sem=cfg.tau_sem,
        lambda_neg=float(getattr(cfg, "planning_lambda_neg", 0.0)),
    )
    danger_head = DangerHead(
        m=context.cfg.danger_dim,
        prior_var=context.env.cfg.learner.hazard_prior_var,
        lr=context.env.cfg.learner.hazard_lr,
    )
    shadow = ShadowLearnerModel(
        scorer=scorer,
        danger_head=danger_head,
        attention_L=8,
        rho_H=context.env.cfg.learner.rho_H,
    )
    return InverseShadowPredictor(
        shadow_model=shadow,
        eta_prof=cfg.eta_prof,
        rollout_mode="proxy",
        n_rollout=8,
        update_semantic=True,
        update_risk=cfg.use_risk,
    )


def fit_inverse_posterior(
    context: TaskContext,
    prelearn_examples,
    observation_runs: Iterable[ObservationRun],
    cfg,
) -> InversePosterior:
    predictor = _build_shadow_predictor(context, prelearn_examples, cfg)
    for run in observation_runs:
        for step in run.steps:
            predictor.observe(step)
    diagnostics = predictor.diagnostics()
    summary = PosteriorSummary(
        profile_posterior=dict(diagnostics.get("profile_posterior", {})),
        profile_entropy=float(diagnostics.get("profile_entropy", 0.0)),
        full_action_nll=diagnostics.get("full_action_nll"),
        pick_nll=diagnostics.get("pick_nll"),
    )
    return InversePosterior(predictor=predictor, summary=summary)
