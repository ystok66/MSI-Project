from __future__ import annotations

from .config import MazeScenarioConfig
from .runner import run_block


def main() -> None:
    cfg = MazeScenarioConfig(
        width=13,
        height=13,
        teach_episodes=4,
        eval_same_map_episodes=4,
        eval_new_map_episodes=4,
        trap_density=0.18,
        extra_loop_prob=0.25,
        learner_unknown_penalty=0.1,
        learner_info_bonus=0.45,
        seed=22,
    )
    for tutor_name in ("no_tutor", "always_warn", "inverse_warn"):
        results = run_block(cfg, tutor_name=tutor_name)
        print(f"\n== {tutor_name} ==")
        for key in sorted(results):
            print(f"{key}: {results[key]:.3f}")


if __name__ == "__main__":
    main()
