# Scenario Suite Index

This file keeps the fixed-map suites clearly separated.

## Large Reference Map

- Code name: `HugeRiskyGemMaze_v0`
- Role:
  the large fixed-map reference scenario for end-to-end runtime and long-horizon experiments.

## Legacy Micro Diagnostic Suite

- Code name: `MiniRiskyMazeSuite_v0`
- Human label:
  `Legacy Micro Diagnostic Suite`
- Maps:
  - `MiniRiskGate_v0`
  - `MiniExploreLoop_v0`
  - `MiniWaypointBottleneck_v0`
- Role:
  the original compact mechanism-debug suite for warning / wait / waypoint diagnosis.

## Didactic Tutor Evaluation Suite

- Code name: `TutorDidacticMazeSuite_v1`
- Human label:
  `Didactic Tutor Evaluation Suite`
- Maps:
  - `TutorSafetyScaffoldGate_v1`
  - `TutorAutonomyLoop_v1`
  - `TutorPrincipleDoorTransfer_v1`
- Role:
  the new compact suite for safety shield, pedagogical scaffolding,
  over-help, and success-gated transfer.

## Naming Rule

- `MiniRiskyMazeSuite_v0`
  refers only to the legacy three-map suite.
- `TutorDidacticMazeSuite_v1`
  refers only to the new three-map suite.

Do not merge these two suites in reports unless the report explicitly says so.

