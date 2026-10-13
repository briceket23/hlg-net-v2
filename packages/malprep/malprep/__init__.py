"""malprep: reproducible data preparation for the HLG-Net confirmatory study (protocol Steps 2-5).

Modules, one per protocol step:
  mpidb      Step 2   per-parasite manifest from the YOLO-seg labels and original images
  dedup      Step 2.4 near-duplicate images, slide / image grouping
  crops      Step 3   fixed 256 px native-resolution crops (+ masks), reproducibility check
  qc         Step 3.4 visual QC sheets, decisions, final manifest
  tables     Step 3.5 Table 1 and derived ratios
  scale      Step 4   red-cell diameter ruler (magnification between datasets)
  bbbc041    Step 4   external stage set, Table 5a, exposure statement
  splits     Step 5   grouped rare-bucket outer folds + nested inner validation, verification
  folds      read-side API used by the training code (partition of one fold)
  nih        NIH cell set (transfer study / split-protocol experiment; not used by the stage-only study)
  synthetic  toy data for tests and dry runs
"""
from malcore import STAGES, STAGE_NAMES, SPECIES  # noqa: F401  (single source of truth)

__version__ = "2.0.0"
