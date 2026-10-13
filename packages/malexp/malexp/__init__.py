"""malexp: training, nested selection and evaluation for the HLG-Net confirmatory study (protocol Steps 6-12).

Modules:
  config      arm definitions (one YAML per arm, one boolean per component), tuning grids   Step 6
  models      backbone -> stage head [+ species head] [+ gate -> bounded boost] [+ M-prob]  Step 6
  losses      CE, logit-adjusted CE, LDAM-DRW, C-HMCNN MCLoss, gate BCE                     Step 6
  data        crops dataset, augmentation, samplers, held-out guard                         Step 7
  train       one run: fit on inner_train/inner_val, finalize -> predictions + meta.json     Steps 7, 11
  select      nested selection rules, B*, selection log                                     Step 9
  runner      fold-by-fold execution of the matrix, resumable, divergence rule              Steps 9-10
  queue       runs/queue.csv (tier, repeat, fold, randomised arm order)                     Step 10
  preflight   one bf16 step of every arm variant on the device                              Step 8
  determinism same-seed rerun check                                                        Step 1
  complexity  params, MACs, CPU latency                                                    Step 13
"""
from malcore import STAGES, SPECIES, RARE, EARLY  # noqa: F401

__version__ = "2.0.0"
