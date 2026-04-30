# one_hint_tutor

`one_hint_tutor` is a standalone protocol package for the one-shot hint experiment.

It intentionally does not reuse `SparseTutorAgent` as the main control loop.

Current first-pass implementation:

- prelearn from a small supervised set
- frozen observation queries for inverse fitting
- one hard teach query with a 20-option menu
- one-shot hint planning from a public inverse posterior
- direct post-teach eval on derived text-to-color examples
- baseline suite:
  - `no_tutor_unlimited`
  - `no_tutor_T7`
  - `random_hard_hint_T6`
  - `tutor_unlimited`
  - `tutor_T6`

Important current simplifications:

- the inverse posterior reuses the existing `InverseShadowPredictor`
  architecture rather than a new SMC implementation
- the planning rollout is profile-mixture beam/MC over the fitted shadow model
- wrong options are removed after reveal to avoid repeated attempts
- refresh is disabled in this protocol
- risk support is wired into menus, metrics, and utility, but this package is
  still primarily aimed at the no-risk first pass
