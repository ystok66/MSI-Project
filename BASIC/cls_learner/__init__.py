"""
cls_learner — CLS (Complementary Learning Systems) three-layer architecture.

Layer 1 (Cortex):  Slow, generalizable concept learning (wraps ns_concept).
Layer 2 (HPC):     Fast, episode-internal memory (DG-CA3-CA1).
Layer 3 (Control): PFC-BG-Cerebellum beam search, selection, execution.
"""
from cls_learner.agent import CLSAgent
from cls_learner.config import CLSConfig
