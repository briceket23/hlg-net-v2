# Architecture: one package per study stage

```
protocol step        package       module                  reads                     writes
-------------------  ------------  ----------------------  ------------------------  --------------------------------
1 provenance         malcore       hashing, provenance, io (everything)              digests, meta fields
2 manifest           malprep       mpidb, dedup            raw labels + images       data/manifests/cells.csv, images.csv
3 crops + QC         malprep       crops, qc, tables       cells.csv                 data/crops, crops.csv, cells_final.csv, Table 1
4 external set       malprep       bbbc041, scale          BBBC041 json + images     bbbc041_cells.csv, Table 5a, exposure stmt
5 splits             malprep       splits, folds           cells_final.csv           splits/<unit>/outer_r*.json, digests.txt
6 arms               malexp        config, models, losses  configs/arms/*.yaml       -
7 one run            malexp        data, train             partitions, crops         runs/.../best.pt, log.csv, meta.json, preds_*
8 verification       malexp        preflight, determinism  -                         determinism.json
9-10 matrix          malexp        select, runner, queue   splits, selection.csv     runs/<unit>/queue.csv, selection.csv
11 completeness      malanalysis   loaders                 preds_*, meta.json        check report
12 external eval     malanalysis   external                preds_bbbc041             external_*.csv, Table 5b
13 metrics           malcore/malanalysis  metrics, endpoints  preds_test            per_fold_metrics.csv, pooled_per_repeat.csv
14 statistics        malanalysis   stats, report           per-fold tables           stats_log.json
17 exhibits          malanalysis   tables, figures         everything above          table*.tex/csv, fig*.pdf
```

Dependency direction: `malcore` ← `malprep` ← `malexp`; `malanalysis` depends on `malcore` only (it
reads files, never models), so the analysis machine does not need PyTorch.

Invariants enforced in code and tests:

- test partitions and BBBC041 are reachable only through `HeldoutGuard` inside `finalize()`;
- training/tuning modules contain no BBBC041 reference (grep test + CI step);
- every run asserts the split and manifest digests before reading a cell;
- only the selected candidate of each (arm, fold) is finalized, so non-selected configs never produce
  test predictions;
- arm YAMLs are verified against `configs/arms/digests.txt` before any run;
- the analysis refuses to start until the Step 11 completeness check passes.
