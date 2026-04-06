"""Quick test: generate and visualize room_grid maps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.envs.benchmark_generator import generate_benchmark_map
from src.envs.pedagogical_grid import PedagogicalGridEnv

for diff in ["easy", "medium", "hard"]:
    for seed in range(3):
        gm, cfg = generate_benchmark_map("room_grid", seed, diff)
        H, W = gm.height, gm.width
        print(f"\n=== room_grid  diff={diff}  seed={seed}  ({H}x{W}) ===")
        print(f"    start={gm.agent_start}  object={gm.object_spawn}  target={gm.target_pos}")
        print(f"    doors={gm.door_positions}  max_steps={cfg.max_steps}  risk_budget={cfg.risk_budget}")

        # Visualize
        for r in range(H):
            row = ""
            for c in range(W):
                ct = gm.cell_types[r, c]
                if (r, c) == gm.agent_start:    row += "S "
                elif (r, c) == gm.object_spawn: row += "O "
                elif (r, c) == gm.target_pos:   row += "G "
                elif ct == 1: row += "# "
                elif ct == 4: row += "D "
                elif ct == 2: row += "! "
                elif ct == 3: row += "$ "
                else:         row += ". "
            print(f"    {row}")

# Run a quick episode
print("\n=== Running 1 episode with no_teacher ===")
gm, cfg = generate_benchmark_map("room_grid", 42, "easy")
env = PedagogicalGridEnv(
    grid_map=gm,
    max_steps=cfg.max_steps,
    initial_risk_budget=cfg.risk_budget,
    prior_risk_mean=cfg.prior_risk_mean,
    prior_risk_var=cfg.prior_risk_var,
    search_budget=cfg.search_budget,
    seed=0,
)
obs, info = env.reset(seed=0)
done = False
steps = 0
while not done:
    obs, reward, term, trunc, info = env.step(0)  # WAIT
    done = term or trunc
    steps += 1
    if steps > cfg.max_steps + 5:
        break

status = "SUCCESS" if info.get("object_delivered") else ("DEATH" if term else "TIMEOUT")
print(f"    steps={steps}  status={status}  reward={reward:.2f}")
print("\n=== room_grid OK ===")
