from __future__ import annotations

import argparse

from .candidate_dataset import build_candidate_dataset_rows, write_candidate_dataset
from .config import OneHintConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidate-level oracle dataset rows.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hint-families", default="free", help="Comma-separated candidate families.")
    parser.add_argument("--teach-menu-size", type=int, default=20)
    parser.add_argument("--max-attempts-main", type=int, default=5)
    parser.add_argument("--prelearn-profile", default="4")
    parser.add_argument("--menu-difficulty-mode", default="rank_stratified", choices=["default", "rank_stratified"])
    parser.add_argument("--teach-probe-mode", default="initial_rank", choices=["initial_rank", "unlimited_tau"])
    parser.add_argument("--target-initial-rank-min", type=int, default=5)
    parser.add_argument("--target-initial-rank-max", type=int, default=12)
    parser.add_argument(
        "--utility-mode",
        default="advantage_delta",
        choices=["legacy", "success_gated", "band_delta", "delta_vs_no_tutor_bonus", "advantage_delta"],
    )
    parser.add_argument("--eval-aware", action="store_true")
    args = parser.parse_args()

    families = tuple(part.strip() for part in args.hint_families.split(",") if part.strip())
    cfg = OneHintConfig(
        seed=int(args.seed),
        hint_mode="combined" if len(families) > 1 else (families[0] if families else "none"),
        hint_families=families or ("free",),
        teach_menu_size=int(args.teach_menu_size),
        max_attempts_main=int(args.max_attempts_main),
        prelearn_profile=str(args.prelearn_profile),
        menu_difficulty_mode=str(args.menu_difficulty_mode),
        teach_probe_mode=str(args.teach_probe_mode),
        target_initial_rank_min=int(args.target_initial_rank_min),
        target_initial_rank_max=int(args.target_initial_rank_max),
        utility_mode=str(args.utility_mode),
        eval_aware=bool(args.eval_aware),
    )
    rows = build_candidate_dataset_rows(
        task_id=str(args.task),
        cfg=cfg,
        seed=int(args.seed),
        families=families,
    )
    write_candidate_dataset(rows, args.out)


if __name__ == "__main__":
    main()
