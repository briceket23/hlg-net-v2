"""malanalysis: the one script that reads the prediction files and writes every number in the paper
(protocol Steps 11-14 and 17). It never trains, never reads crops, and never selects anything.

Modules:
  loaders     run discovery, prediction loading, the per-fold B* arm, completeness check     Step 11
  endpoints   per-fold and pooled-per-repeat metrics, species macro-F1                        Step 13
  stats       corrected resampled t-test, Holm, exact sign test, grouped bootstrap, power     Step 14
  external    BBBC041 zero-shot evaluation per model, by subset, paired contrast, bootstrap   Step 12
  tables      CSV + LaTeX exhibits (Tables 2-5)                                               Step 17
  figures     paired differences, confusion matrices, reliability diagrams (Figures 2-4)      Step 17
  report      run_analysis(): orchestration, stats_log.json, decision-rule wording            Steps 13-14
  plan        timestamped, hashed analysis plan (gate G4)                                     Step 9
  synthetic   a synthetic run tree for testing the pipeline end to end
"""
from malcore import STAGES, STAGE_NAMES, RARE  # noqa: F401

__version__ = "2.0.0"
