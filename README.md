# HLG-Net confirmatory study — `hlg-net-v2`

Code for the stage-only confirmatory study of HLG-Net: *does a supervised rare-stage gate with a
bounded logit boost (M-full) improve four-class stage macro-F1 on annotated malaria parasite crops over
the best matched rebalanced baseline B\*?* Training and test data are MP-IDB (four species, stages
ring / trophozoite / schizont / gametocyte); BBBC041 is a zero-shot external set. The operating
protocol (revision 2, after the external methodological audit) is in [`docs/protocol.md`](docs/protocol.md);
this repository implements every numbered step of it as code, tests and generated artefacts.

The repository is a monorepo of four installable packages, one per study stage:

| Package | Protocol steps | What it does |
|---|---|---|
| [`packages/malcore`](packages/malcore) | 1, 13 | shared constants (stage indices, rare group L), SHA-256 digests, portable I/O, provenance (git commit + dirty flag, GPU, versions), numpy metrics |
| [`packages/malprep`](packages/malprep) | 2–5 | MP-IDB per-parasite manifest, near-duplicate grouping, 256 px native crops, visual QC, Table 1, BBBC041 set and Table 5a, red-cell magnification ruler, rare-bucket grouped outer folds + nested inner validation, gates G1–G2, freeze lock, NIH patient splits (thesis only) |
| [`packages/malexp`](packages/malexp) | 6–12 | arm configs (one YAML per arm, one flag per component), network (backbone → stage head, species head, gate → bounded boost, M-prob), losses (CE, logit-adjusted, LDAM-DRW, C-HMCNN MCLoss, gate BCE), shared training recipe with held-out guard, temperature scaling, prediction files, nested selection of B\*, resumable fold runner, queue, preflight, determinism and complexity checks |
| [`packages/malanalysis`](packages/malanalysis) | 11–14, 17 | completeness check, per-fold and pooled endpoints, corrected resampled t-test, Holm, sign test, grouped bootstrap, BBBC041 external evaluation, tables (LaTeX + CSV), figures, `stats_log.json`, timestamped analysis plan |

`malanalysis` needs no PyTorch: it reads prediction files only. See [`docs/architecture.md`](docs/architecture.md)
for the module map and the invariants that are enforced in code.

## Layout

```
configs/data.yaml            data-preparation settings (frozen at G1/G2; fill in label_source, see below)
configs/arms/                B0..B4 YAMLs, grids.yaml, derived.yaml, digests.txt   (gate G4)
configs/analysis.yaml        rho, alpha, Delta_min, sigma_d range, bootstrap, families   (gate G4)
splits/{slide,image,nih}/    frozen split files + digests (12 Oct 2026 freeze)          (gate G2)
data/                        raw/ (not tracked), manifests/ + tables/ (copy the 34 frozen files here), crops/ (regenerated)
runs/<unit>/                 queue.csv, selection.csv, <arm>/<backbone>/r{r}_f{k}/c{i}/{best.pt, log.csv, meta.json, preds_*.parquet}
analysis/                    analysis_plan.md (+ .sha256), stats_log.json, table*.tex|csv, fig*.pdf, complexity.csv
logs/deviations.md           Step 16 deviations log (reproduced in the supplement)
docs/                        protocol.md, architecture.md, review_findings.md
tools/                       install.sh, dry_run.sh, run_matrix.sh
.github/workflows/ci.yml     tests on CPU, arm-digest check, "training never reads BBBC041" grep
```

## Install

```bash
git clone https://github.com/briceket23/hlg-net-v2 && cd hlg-net-v2
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel; on the GPU machine install the CUDA build instead
python -m pip install -r requirements/dev.txt
python -m pip install -e packages/malcore -e packages/malprep -e packages/malexp -e packages/malanalysis
python -m pytest            # 61 tests, CPU, ~3 min (Step 8 tests, pipeline tests, analysis on a synthetic 180-run tree)
```

`make install`, `make test`, `make freeze-env` wrap the same commands. Pin the environment on the GPU
machine with `python -m pip freeze > env/freeze.txt` (Step 1).

## Run order (one command per protocol step)

All commands run from the repository root. Raw data go to `raw_data/` as described in
`configs/data.yaml` (`mp_idb/images`, `mp_idb/labels`, `BBBC041/malaria`, `cell_images`).

### Steps 2–5: data (`malprep`)

| Step | Command | Output |
|---|---|---|
| 2 manifest | `python -m malprep manifest` | `data/manifests/cells.csv`, `exclusions.csv`, `images.csv` |
| 2.4 duplicates, groups | `python -m malprep dedup` | `group_slide`, `group_image`, `near_duplicates.csv` |
| 3 crops | `python -m malprep crops` | `data/crops/mpidb/{img,mask}/`, `crops.csv` |
| 3.4 QC | `python -m malprep qc-sheets`, `qc-mark --stages S,G --decision ok --reviewer XX`, `qc-apply` | `data/qc/qc.csv`, `cells_final.csv` |
| 3.5 Table 1 | `python -m malprep tables` | `data/tables/table1_data.{csv,md}`, `derived_ratios.json` |
| 4 ruler + external set | `python -m malprep rbc-scale`, `python -m malprep bbbc041` | `bbbc041_cells.csv`, crops, `table5a_external_data.csv`, exposure statement |
| 5 splits | `python -m malprep splits --unit slide` and `--unit image` | `splits/<unit>/outer_r{1,2,3}.json`, `digests.txt`, `fold_coverage_<unit>.csv` |
| G1–G2 | `python -m malprep verify`, `python -m malprep gate`, `python -m malprep freeze` | `data/manifests/FROZEN.json` (locks every writer) |

The data were frozen on 12 Oct 2026 (34 files; 1,435 MP-IDB parasites R 1,283 / T 81 / S 41 / G 30 on
17 slides; 2,452 BBBC041 cells). The split files are in this repository; copy the frozen
`data/manifests/` and `data/tables/` next to them and run `python -m malprep gate`. The gate now fails
while `label_source` in `configs/data.yaml` still holds a placeholder: record the source of the label
files and re-freeze with `python -m malprep freeze --amend --reason "label_source recorded"`
(the amendment is stored in `FROZEN.json`; see Deviation 5).

### Steps 6–10: training (`malexp`)

| Step | Command | Notes |
|---|---|---|
| 6 arms (G4) | `python -m malexp arms-export` | writes `configs/arms/*.yaml` + `digests.txt`; every run verifies them |
| 8 code verification (G3) | `python -m pytest` | all Step 8 tests, including the nested smoke run of the 13 arms |
| 8 device check | `python -m malexp preflight` | one bf16 forward/backward/inference step of all 37 arm variants on the GPU |
| 8 GPU pilot | `python -m malexp pilot` | one M-full run on the inner split of (r1, k0): batch, epoch time, memory; scores never reported |
| 1 determinism | `python -m malexp determinism --arm M-full --max-epochs 10` | two same-seed runs; spread ≤ 0.005 expected, recorded otherwise |
| 9 analysis plan (G4) | `python -m malanalysis plan` | `analysis/analysis_plan.md` + SHA-256; commit with a signed tag before Step 10 |
| 10 matrix | `python -m malexp run --unit slide --tiers 1 2 3` | writes `runs/slide/queue.csv`, trains fold by fold, resumable |
| 10 sensitivity | `python -m malexp run --unit image --tiers 1 2` | image-grouped splits |
| 10 replication | `python -m malexp run --unit slide --tiers 1 2 --backbone efficientnet_b0.ra_in1k --input-size 224 --runs runs/slide_effb0` | Tier 4 |
| 13 complexity | `python -m malexp complexity` | params, MACs, CPU latency (batch 1, 4 threads, 50 + 200 runs) → `analysis/complexity.csv` |
| status | `python -m malexp status` | trained / finalized runs against the queue |

`tools/run_matrix.sh` chains these in protocol order. Within a fold the runner trains B0 first, then
B1–B4 (2 configs each), selects B\* on inner validation (tie ≤ 0.005 → simpler: B2 < B1 < B4 < B3),
then B5, B6, M-aux, M-full (and M-prob with `--with-prob`) built on B\*, then the three ablations from
the selected M-full. Only the selected candidate of each arm is finalized, so non-selected configs never
produce test predictions. A run that diverges (non-finite loss, or inner-val macro-F1 below B0 − 0.10)
is rerun once with the same seed and the attempt is kept.

### Steps 11–14, 17: analysis (`malanalysis`)

| Step | Command | Output |
|---|---|---|
| 11 completeness (G5/G6) | `python -m malanalysis check` | every (arm, r, k) has its three files; test cells = manifest per repeat; digests match |
| 13–14, 17 | `python -m malanalysis run` | `analysis/stats_log.json`, `per_fold_metrics.csv`, `pooled_per_repeat.csv`, Tables 2–5 (`.tex` + `.csv`), Figures 2–4, `external_*.csv`, selection summary |

The primary contrast is M-full − B\* on per-fold macro-F1 with the corrected resampled t-test
(ρ = 0.25, df = 14) and the decision rule of Step 14, whose wording is used verbatim. Family A (pooled
schizont / gametocyte F1 with grouped-bootstrap CIs, balanced accuracy, external macro-F1), Family B
(ladder) and Family C (M-aux and ablations) are Holm-adjusted within family. Sensitivity analyses
(grouped bootstrap, sign test, repeat 1 only, EfficientNet-B0 replication, image-grouped splits) are
reported but never used for decisions. The script refuses to run until the completeness check passes.

## What one run does (Step 7)

`Trainer(cfg, data_cfg, unit, r, k, run_dir)` asserts the split and manifest digests, reads
`inner_train` / `inner_val` through `malprep.folds.get_partition`, computes class priors, LDAM margins,
DRW weights and sampler weights from `inner_train` only, and sets the gate BCE `pos_weight` from the
rare-stage prevalence *as sampled*. AdamW (wd 0.05, none on bias/norm), head lr = 10 × backbone lr,
5-epoch warm-up then cosine to 1 %, 60-epoch cap, grad clip 1.0, bf16 on CUDA, early stopping on
inner-val macro-F1 (patience 15, min delta 0.002). Schedules: B3 runs the fixed budget with DRW from
80 % of the cap and selects its checkpoint inside that phase; B4 loads the fold's selected B0 and trains
only phase 2 (frozen backbone, new head, q = 1 sampler, 15 epochs); M-prob uses the fixed budget; arms
derived from a B4 B\* keep gate and boost frozen in phase 2. `finalize()` loads the best checkpoint,
fits one temperature on inner validation, and only then builds the test and BBBC041 loaders, writes
`preds_test`, `preds_innerval`, `preds_bbbc041` (one row per cell: logits, calibrated probabilities,
prediction, gate, species head for B5, `image_group`, `config_sha`, `split_sha`, `manifest_sha`,
`commit`) and `meta.json` with every Step 1 field.

## Arms (Step 6), exactly as coded

| Arm | Objective / sampler | Grid (2 configs per fold) |
|---|---|---|
| B0 | CE, instance sampler | lr {3e-4, 1e-3} |
| B1 | CE, class-balanced sampler q = 0.5 | lr |
| B2 | logit-adjusted CE, τ = 1; prediction = argmax raw logits | lr |
| B3 | LDAM (cosine head, s = 30, max margin 0.5) + DRW from 80 % of the cap (β = 0.9999), fixed budget | lr |
| B4 | cRT on the fold's selected B0; phase 2 head lr | {1e-3, 1e-2} |
| B5 | B\* + λ_sp · CE(species) | λ_sp {0.3, 1.0} |
| B6 | B\* rebalancing + C-HMCNN MCLoss; a loss-based B\* (B2/B3) switches to the B1 sampler | lr |
| M-aux | B\* + gate (MLP d→64→1, GELU, dropout 0.1, gradient into the backbone) + λ_g · BCE | λ_g {0.15, 0.5} |
| M-full | M-aux + boost b·g on the two rare logits, b = 5σ(β), β₀ = −2.94 (b₀ ≈ 0.25), before the objective | λ_g {0.15, 0.5} |
| M-prob (optional) | gate as p(rare); log p(c) = log p(group) + log softmax within group; fixed budget | λ_prior {0, 1} |
| ABL-noaux / ABL-detach / ABL-coordattn | one flag changed from the fold's selected M-full | none |

## Tests (Step 8)

`python -m pytest` runs 61 tests on CPU: metrics against scikit-learn (100 random sets), ECE against a
reference, each flag toggles exactly one thing, boost off ⇔ M-aux, boost starts at 0.25 and is applied
before logit adjustment, gate gradient reaches the backbone iff not detached, cRT phase 2 freezes gate
and boost, τ = 0 is CE, zero LDAM margin is scaled cosine CE, shift invariance, C-HMCNN consistency and
float32 loss, M-prob sums to 1, sampler frequencies and sampled prevalence, selection tie rules, B6
rule and fixed budgets, arm YAML export/verification, idempotent selection log, held-out guard,
per-sample augmentation streams, "training code never touches BBBC041", every arm overfits 64 crops,
the nested smoke run of all 13 arms with meta/prediction provenance and resume, digest mismatch blocks
training, 37-variant preflight, complexity and determinism commands; the malprep pipeline tests on a
synthetic toy dataset (manifest, duplicates, crops, QC, Table 1, BBBC041, splits, gate, freeze, lock);
and the analysis on a synthetic 180-run tree (corrected t-test, decision wording, Holm, sign test,
power statement, grouped bootstrap, completeness, every exhibit).

## Gates

| Gate | Command / artefact |
|---|---|
| G1 data frozen | `python -m malprep gate` → `freeze`; `FROZEN.json`, label provenance, exposure statement |
| G2 splits frozen | `splits/<unit>/digests.txt`, `python -m malprep verify` |
| G3 code verified | `python -m pytest` green in CI on the tagged commit (`v1-code-verified`) |
| G4 configs and plan frozen | `configs/arms/digests.txt`, `analysis/analysis_plan.sha256`, tag `v2-frozen` |
| G5 Tier 1 complete | `python -m malanalysis check` passes for M-full, B\*, M-aux |
| G6 all tiers complete | `python -m malanalysis check` passes for every arm run |
| G7 numbers from code | every number traceable to `analysis/stats_log.json` or a generated table |

## Status (13 Oct 2026)

- Data frozen (G1/G2, 12 Oct) — frozen manifests still to be copied into `data/`; `label_source` to be
  recorded and re-frozen with `--amend`.
- Code rebuilt and verified on CPU (this repository); the GPU checks (`preflight`, `determinism`,
  `pilot`) must be repeated on the training machine before tagging `v1-code-verified` / `v2-frozen`.
- The runs finished under `v2-frozen-1` on fold r1/k0 are discarded and rerun (Deviations 2–3 in
  [`logs/deviations.md`](logs/deviations.md)); no test prediction has been opened.
- Still to do: the matrix (Tiers 1–5), `complexity`, the analysis, the exhibits.

See [`docs/review_findings.md`](docs/review_findings.md) for what the rebuild changed and why.
