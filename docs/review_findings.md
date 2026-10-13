# Review of the uploaded packages (13 Oct 2026)

Inputs: `malprep_data_preparation.zip` (malprep 1.0.0 + frozen split files + `test_step8.py` of
`v2-frozen-1`), `malexp_training.zip` (malexp 1.0.0, a copy of malprep, `test_step8.py` of
`v2-frozen`), and the protocol (`docs/protocol.md`, revision 2). Everything was re-read line by line
against the protocol; the findings below drove the rebuild.

## What was already in place and was kept

- malprep: manifest from polygon labels, two-stage near-duplicate test, 256 px native-window crops with
  byte-identical reproduction, QC sheets and command-line decisions, Table 1, red-cell ruler, BBBC041
  manifest and Table 5a, rare-bucket outer folds with best-of-N balancing and repeat-overlap penalty,
  nested inner validation, leakage verification, freeze lock, NIH patient grouping. All 20 tests pass.
- malexp: one flag per component, boost before adjustment, gate pos_weight from the sampled
  prevalence, held-out guard, temperature scaling, nested selection with the tie rules, 37-variant
  preflight, float32 MCLoss (Deviation 1), resumable runner.
- The frozen split files (`splits/slide`, `splits/image`, `splits/nih`) are imported unchanged; their
  digests match the ones in the zip.

## Defects found and fixed (see `logs/deviations.md` for the protocol consequences)

| # | Where | Finding | Fix |
|---|---|---|---|
| 1 | malexp `train.fit` | B4 retrained its own phase 1 with the default lr; protocol: phase 1 = the fold's selected B0 | runner trains B0 first; B4 loads its checkpoint; only phase 2 trained |
| 2 | malexp `train._phase` | B3 could early-stop before DRW and select a phase-1 checkpoint | fixed budget, DRW phase 2, selection inside phase 2 |
| 3 | malexp `train.fit` | M-prob early-stopped; protocol: fixed budget | `fixed_budget` flag |
| 4 | malexp `models.head_parameters` | cRT phase 2 trained the gate and boost of derived arms | `freeze_gate()`; phase-2 head parameters exclude gate/boost |
| 5 | malexp `config.derive_from_bstar` | B6 on B2/B3 lost the rebalancing (MCLoss replaced the objective) | B1 sampler (q = 0.5) with MCLoss, per the Step 6 table |
| 6 | malexp `data.Transform` | one RNG copied into every worker: identical augmentation sequences across workers | per-sample generator seeded from (seed, worker seed, index, counter) |
| 7 | malexp `train._write_meta` / `_write_preds` | missing Step 1 / Step 11 fields: dirty flag, manifest/split SHA, driver, start/end, selection record; preds lacked `image_group`, `split_sha`, `manifest_sha`, `commit` | all written; tested |
| 8 | malexp `train` | no assertion that splits and manifest match `digests.txt` (Step 7 run procedure 1) | `check_split_digests` at every run start; tested |
| 9 | malexp `select.SelectionLog` | duplicate rows on resume; no backbone column (replication would be ambiguous) | idempotent on key; backbone, run id, tier, divergence, reruns recorded |
| 10 | malexp runner | Step 10 divergence rule not implemented | rerun once, attempt kept as `*_diverged_attempt1.*` |
| 11 | malexp | no `queue.csv`, no species output for B5, no complexity / determinism commands | `queue`, `species_pred`, `complexity`, `determinism` |
| 12 | malprep gate | `label_source` placeholder `<fill in>` passed the G1 check | placeholder regex; `freeze --amend --reason` for a logged re-freeze |
| 13 | malprep `folds` | a second `CropDataset` duplicated malexp's | removed; `image_group` column added to partitions |
| 14 | both | `STAGES` defined three times; hashing / IO / provenance duplicated | `malcore` |
| 15 | packaging | malexp zip carried a stale copy of malprep and its older test file | monorepo of four packages, one test command, CI |

## Not changed on purpose

- The data recipe, the crop rule, the split algorithm and the split files: they are frozen (G1/G2).
- The shared training recipe (AdamW 0.05, warm-up 5, cosine to 1 %, batch 64, bf16, patience 15,
  min delta 0.002), the grids, the tie rules, the gate architecture and the boost parametrisation.
- `configs/data.yaml` is the uploaded one. Its SHA-256 (`f713c458…`) differs from the one recorded in
  the frozen split provenance (`932ecebb…`): the file was edited after the freeze (the `label_source`
  line). Fill in the placeholder and re-freeze with `--amend`.

## Still to do on the GPU machine (not possible here)

- `python -m malexp preflight` and `python -m malexp determinism` on the GPU (Step 1 / Step 8 checks).
- Re-run fold r1/k0 and the rest of the matrix under the new tag (`tools/run_matrix.sh`).
- `python -m malexp complexity` on the named CPU for Table 3.
