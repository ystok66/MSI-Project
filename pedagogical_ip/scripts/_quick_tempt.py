import sys; sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
FAMILY = "goal_preference_temptation_entanglement_lattice"
runner = LatticeV2Runner()
def run_ep(seed, fm, ucfg=None):
    try:
        s = runner.reset(seed=seed, difficulty="hard", scenario_family=FAMILY,
            robot_belief_mode=True, intervention_family_mode=True,
            item_drop_enabled=True, belief_planning_mode=True,
            latent_mode=True, patch_radius=2, prefix_horizon=5,
            factor_mode=fm, user_cfg=ucfg)
        while not s.done: s = runner.step(s)
        m = runner.get_metrics(s)
        return bool(m["survived"])
    except: return False

ucfg = {"lure_strength": 0.95, "tempt_offset_z": 0.90, "deadline_slack_final": 1.02}
print("--- tempt+tight (fair dispatch) ---")
for fm in ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
    s = np.mean([run_ep(i, fm, ucfg) for i in range(30)])
    d = "" if fm == "FULL" else f" D={s - full_s:+.3f}"
    if fm == "FULL": full_s = s
    print(f"  {fm:12s}: surv={s:.3f}{d}")
