"""Quick smoke test for funnel_trap generator."""
import sys
sys.path.insert(0, ".")
from src.envs.scenario_families import generate_scenario

for diff in ["easy", "medium", "hard"]:
    for seed in [0, 1, 42]:
        gm, cfg, meta, sc = generate_scenario("funnel_trap", seed, diff, latent_mode=True)
        print(f"{diff} s{seed}: {gm.height}x{gm.width} t_max={cfg.max_steps} "
              f"segs={len(meta.segments)} safe={meta.shortest_safe} any={meta.shortest_any} "
              f"decision={sc.decision_points} commit={sc.commitment_points} "
              f"trap_corr={sc.trap_corridor_stage2}")

# Run one episode through the runner
from src.envs.lattice_v2_runner import LatticeV2Runner
runner = LatticeV2Runner()
s = runner.reset(seed=42, scenario_family="funnel_trap", latent_mode=True,
                 difficulty="medium", tutor_mode="none", warning_mode="none")
steps = 0
while not s.done:
    runner.step(s)
    steps += 1
m = runner.get_metrics(s)
print(f"\nEpisode: steps={m['steps']} goal={m['reached_goal']} survived={m['survived']}")
print("OK")
