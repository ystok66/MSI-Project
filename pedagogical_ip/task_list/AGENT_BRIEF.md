# AGENT BRIEF — pedagogical_ip

## Project goal

This project studies pedagogical inverse planning:
when should a robot/teacher intervene to help a bounded-rational learning agent, and how?

The current codebase already has:
- a Gymnasium environment,
- an internal bounded-rational learner,
- particle-based teacher inference,
- symbolic/RSA-style warning,
- benchmark map families,
- lattice_v2 warning-vs-door experiments.

But the proposal aims at a stronger model:
- POMDP-style agent representation,
- belief over agent's belief (nested ToM),
- sequential Bayesian inverse planning / bounded action prediction,
- pragmatic informing,
- structural intervention,
- optional item-drop / shield intervention,
- better evaluation metrics tied to learning-vs-help tradeoff.

## Main files to read first

1. `proposal.md`
2. `README.md`
3. `README_lattice_v2.md`

Then inspect:
- `src/envs/`
- `src/agents/`
- `src/teachers/`
- `configs/`
- `tests/`

## External reference repos

Located in `external_refs/`:

### 1. Minigrid
Use for:
- environment API design
- partial-observation environment structure
- clean task/environment separation

Do NOT:
- rewrite the whole project into Minigrid
- replace the current environment unless clearly justified

### 2. pomdp-py
Use for:
- belief/state/action/observation abstractions
- planner/inference interface separation
- robot belief over agent-belief refactor ideas

Do NOT:
- fully port the project into pomdp-py
- introduce a huge dependency-heavy rewrite

### 3. pypragmods
Use for:
- cleaner pragmatic warning semantics
- utterance → listener belief update abstraction
- replacing ad hoc warning logic with a simplified RSA-style module

Do NOT:
- overcomplicate language
- add deep recursive pragmatics unless minimal and justified

### 4. ave
Use for:
- baseline ideas for proactive assistance
- framing help-vs-autonomy tradeoff

Do NOT:
- force empowerment into the core model unless useful as a baseline

### 5. pddlgym
Use for:
- future compositional-goal extensions
- symbolic planning structure ideas

Do NOT:
- migrate current experiments to PDDL right now

## Current scientific gap

The current implementation is strong as a prototype, but still weaker than the proposal in three places:

1. Agent/robot mental-state modeling is not yet a clean POMDP + nested-belief abstraction.
2. Planning/inverse-planning interface is still too custom and heuristic.
3. Warning semantics work, but are not yet a clean modular pragmatic listener/speaker design.

## Priority task

Please do NOT rewrite the project.

Instead, produce a minimal-diff improvement plan for these three targets:

### Target A — Environment/API cleanup
Use Minigrid only as inspiration.
Goal: make environment, agent state, observation, and intervention interfaces cleaner and more modular.

### Target B — Belief / teacher refactor
Use pomdp-py as inspiration.
Goal: separate:
- world state
- agent belief
- robot belief about agent belief
- predictive action model

### Target C — Warning semantics refactor
Use pypragmods as inspiration.
Goal: replace the current lane-warning logic with a simple modular pragmatic warning model that still matches current experiments.

## Constraints

- Prefer minimal diffs.
- Preserve current experiment scripts if possible.
- Do not break existing benchmark behavior.
- Keep the current lattice_v2 experiments runnable.
- Prefer Python-only solutions.
- Avoid large framework migration.
- Focus on readable structure and research usefulness, not software perfection.

## Expected output format

Please return:

1. A concise diagnosis of the current codebase
2. A ranked refactor plan (small / medium / large changes)
3. A list of exact files to modify first
4. For each modification:
   - why it helps scientifically
   - why it is minimal
   - what external repo inspired it
5. Then implement only the smallest high-value changes first