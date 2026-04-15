============================================================
# Q1: Query Source
============================================================

Task file: F:\SCAI\Learning-agent\BASIC\cls_learner\data\000001.txt
Support count: 14
Query count: 10
Grammar nouns: {'dax': 'BLUE', 'lug': 'YELLOW', 'blicket': 'PURPLE', 'kiki': 'GREEN'}
Grammar colors: ['BLUE', 'GREEN', 'PURPLE', 'YELLOW']
Grammar rules: [(['u1', 'fep', 'u2'], ['[u1]', '[u2]']), (['x1', 'tufa'], ['[x1]', '[x1]', '[x1]']), (['x1', 'gazzer'], ['[x1]', '[x1]']), (['u1', 'x1'], ['[u1]', '[x1]'])]

## All Queries from 000001.txt:
  Q0: IN: dax tufa gazzer  OUT: BLUE BLUE BLUE BLUE BLUE BLUE
       output length = 6
  Q1: IN: dax gazzer gazzer  OUT: BLUE BLUE BLUE BLUE
       output length = 4
  Q2: IN: kiki fep lug  OUT: GREEN YELLOW
       output length = 2
  Q3: IN: dax tufa  OUT: BLUE BLUE BLUE
       output length = 3
  Q4: IN: kiki kiki fep kiki gazzer  OUT: GREEN GREEN GREEN GREEN GREEN GREEN
       output length = 6
  Q5: IN: kiki  OUT: GREEN
       output length = 1
  Q6: IN: kiki fep dax tufa  OUT: GREEN BLUE GREEN BLUE GREEN BLUE
       output length = 6
  Q7: IN: kiki gazzer tufa  OUT: GREEN GREEN GREEN GREEN GREEN GREEN
       output length = 6
  Q8: IN: kiki fep dax  OUT: GREEN BLUE
       output length = 2
  Q9: IN: blicket lug lug gazzer  OUT: PURPLE YELLOW YELLOW PURPLE YELLOW YELLOW
       output length = 6

## Query counts per task file:
  000001.txt: 14 support, 10 query, colors=['BLUE', 'GREEN', 'PURPLE', 'YELLOW']
  000002.txt: 14 support, 10 query, colors=['PINK', 'PURPLE', 'RED', 'YELLOW']
  000003.txt: 14 support, 10 query, colors=['BLUE', 'GREEN', 'PURPLE', 'RED']
  000004.txt: 14 support, 10 query, colors=['BLUE', 'PINK', 'PURPLE', 'YELLOW']
  000005.txt: 14 support, 10 query, colors=['GREEN', 'PINK', 'PURPLE', 'RED']

============================================================
# Q2: Parameter Values Used in Experiments
============================================================

## Default Config:
  n_sup = 14
  n_em = 3
  n_obs_queries = 4
  n_teach_queries = 8
  n_eval_queries = 8
  use_observation_phase = True

## Query Split for 000001.txt (available=10):
  Requested: n_obs=4, n_teach=8, n_eval=8
  Total needed: 20
  OVERFLOW: YES (20 > 10)
  Scaled: n_obs=2, n_teach=4, n_eval=4
  PROBLEM: Only 4 teach queries and 4 eval queries!

## Obs queries:
  obs[0]: dax tufa gazzer -> BLUE BLUE BLUE BLUE BLUE BLUE
  obs[1]: dax gazzer gazzer -> BLUE BLUE BLUE BLUE

## Teach queries:
  teach[0]: kiki fep lug -> GREEN YELLOW
  teach[1]: dax tufa -> BLUE BLUE BLUE
  teach[2]: kiki kiki fep kiki gazzer -> GREEN GREEN GREEN GREEN GREEN GREEN
  teach[3]: kiki -> GREEN

## Eval queries:
  eval[0]: kiki fep dax tufa -> GREEN BLUE GREEN BLUE GREEN BLUE
  eval[1]: kiki gazzer tufa -> GREEN GREEN GREEN GREEN GREEN GREEN
  eval[2]: kiki fep dax -> GREEN BLUE
  eval[3]: blicket lug lug gazzer -> PURPLE YELLOW YELLOW PURPLE YELLOW YELLOW

============================================================
# Q3 & Q4: Divergence Investigation
============================================================

## Initial shadow state:
  Shadow grammar words: ['blicket', 'dax', 'fep', 'gazzer', 'kiki', 'lug']
  Shadow fidelity: exact

## Initial prediction comparison (shadow vs real) on ALL queries:
  Q0 (dax tufa gazzer):
    real:   ['BLUE', 'BLUE', 'BLUE', 'BLUE']
    shadow: ['BLUE', 'BLUE', 'BLUE', 'BLUE']
    agree:  True
  Q1 (dax gazzer gazzer):
    real:   ['BLUE', 'BLUE', 'BLUE', 'BLUE']
    shadow: ['BLUE', 'BLUE', 'BLUE', 'BLUE']
    agree:  True
  Q2 (kiki fep lug):
    real:   ['GREEN', 'YELLOW']
    shadow: ['GREEN', 'YELLOW']
    agree:  True
  Q3 (dax tufa):
    real:   ['BLUE', 'BLUE']
    shadow: ['BLUE', 'BLUE']
    agree:  True
  Q4 (kiki kiki fep kiki gazzer):
    real:   ['GREEN', 'GREEN', 'GREEN', 'GREEN']
    shadow: ['GREEN', 'GREEN', 'GREEN', 'GREEN']
    agree:  True
  Q5 (kiki):
    real:   ['GREEN']
    shadow: ['GREEN']
    agree:  True
  Q6 (kiki fep dax tufa):
    real:   ['GREEN', 'BLUE', 'BLUE']
    shadow: ['GREEN', 'BLUE', 'BLUE']
    agree:  True
  Q7 (kiki gazzer tufa):
    real:   ['GREEN', 'GREEN', 'GREEN', 'GREEN']
    shadow: ['GREEN', 'GREEN', 'GREEN', 'GREEN']
    agree:  True
  Q8 (kiki fep dax):
    real:   ['GREEN', 'BLUE']
    shadow: ['GREEN', 'BLUE']
    agree:  True
  Q9 (blicket lug lug gazzer):
    real:   ['PURPLE', 'YELLOW', 'YELLOW', 'YELLOW']
    shadow: ['PURPLE', 'YELLOW', 'YELLOW', 'YELLOW']
    agree:  True

  ALL AGREE: True

## Object identity check (BUG HUNT):
  OK: shadow_agent is NOT real_agent (independent)
  Real library id: 1265593124608
  Shadow library id: 1265593195520
  Same library: False

  Word 'blicket':
    Real role_counts:   {'EMIT': np.float64(18.185003571174587), 'REPEAT': np.float64(0.06499642882542195), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': 0.0, 'OVER_INFIX': 0.0}
    Shadow role_counts: {'EMIT': np.float64(18.185003571174587), 'REPEAT': np.float64(0.06499642882542195), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': 0.0, 'OVER_INFIX': 0.0}
    Real emit_w:   32.8382
    Shadow emit_w: 32.8382
    Role counts match: True

  Word 'dax':
    Real role_counts:   {'EMIT': np.float64(12.753652309749022), 'REPEAT': np.float64(0.17782188861870193), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': np.float64(0.06852580163227753), 'OVER_INFIX': 0.0}
    Shadow role_counts: {'EMIT': np.float64(12.753652309749022), 'REPEAT': np.float64(0.17782188861870193), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': np.float64(0.06852580163227753), 'OVER_INFIX': 0.0}
    Real emit_w:   23.4089
    Shadow emit_w: 23.4089
    Role counts match: True

  Word 'fep':
    Real role_counts:   {'EMIT': np.float64(0.006622014855231226), 'REPEAT': np.float64(0.14971569091608508), 'SWAP_INFIX': np.float64(0.6585114227247828), 'CONCAT_INFIX': np.float64(14.87574782566212), 'OVER_INFIX': np.float64(0.059403045841781715)}
    Shadow role_counts: {'EMIT': np.float64(0.006622014855231226), 'REPEAT': np.float64(0.14971569091608508), 'SWAP_INFIX': np.float64(0.6585114227247828), 'CONCAT_INFIX': np.float64(14.87574782566212), 'OVER_INFIX': np.float64(0.059403045841781715)}
    Real emit_w:   0.0120
    Shadow emit_w: 0.0120
    Role counts match: True

  Word 'gazzer':
    Real role_counts:   {'EMIT': 0.0, 'REPEAT': np.float64(1.6756994937293324), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': 0.0, 'OVER_INFIX': np.float64(0.07430050627066728)}
    Shadow role_counts: {'EMIT': 0.0, 'REPEAT': np.float64(1.6756994937293324), 'SWAP_INFIX': 0.0, 'CONCAT_INFIX': 0.0, 'OVER_INFIX': np.float64(0.07430050627066728)}
    Real emit_w:   0.0000
    Shadow emit_w: 0.0000
    Role counts match: True

## After teaching one query with feedback:
  Teach query: kiki fep lug
  Outcome: SUCCESS
  Confirms: 1

  After teaching, predictions:
    real:   ['GREEN', 'YELLOW']
    shadow: ['GREEN', 'YELLOW']
    agree:  True

  No feedback update occurred (outcome=SUCCESS)
  'blicket': still matches
  'dax': still matches
  'fep': still matches

============================================================
# DIAGNOSIS SUMMARY
============================================================

## Key Facts:

1. Queries come from: txt files directly (parse_task_file)
   - NOT generated, just parsed from BASIC/cls_learner/data/*.txt
   - Task 000001 has 10 queries total

2. Default config: n_obs=4, n_teach=8, n_eval=8
   - Total needed: 20
   - Available in 000001.txt: 10
   - OVERFLOW: True

3. Shadow divergence = 0 root cause analysis:
   The shadow is reconstructed by:
     a. Creating a new CLSAgent
     b. Calling agent.study(support) to establish vocabulary + structure
     c. Overwriting library counts from snapshot
   Since the support set is IDENTICAL and updates are deterministic,
   the shadow's grammar state IS the real learner's grammar state
   (at the time of snapshot creation).
   
   HOWEVER: during teaching, the REAL learner gets feedback updates
   (differential M-step modifying role_counts/emit_stats), but the
   SHADOW does NOT get updated in sync. The divergence measurement
   only happens BEFORE each teach query, and the shadow is initialized
   just before teaching starts. If no feedback has been applied yet,
   divergence is naturally 0.
   
   The BUG: divergence is measured at the START of each teach query
   but the shadow state is NEVER updated during teaching loops in
   run_phase3.py. The shadow only updates during ShadowTutor.on_select()
   (risk updates) and on_confirm_fail() (hint decisions), but the
   grammar side is not sync'd. Meanwhile the divergence test only
   measures the shadow's INITIAL state.
