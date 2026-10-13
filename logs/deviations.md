# Deviations log (protocol Step 16)

Every change to the protocol or to frozen code after gate G4 is entered here **before** it is run.
Each entry: date · step · what changed and why · whether any test result had been seen.
The paper's supplement reproduces this log in full.

## Deviation 1 — 13 Oct 2026 · Step 6/8 · C-HMCNN MCLoss under CUDA bf16 autocast

During fold r1/k0 of the slide-grouped matrix the first B6 candidate stopped at its first training step:
PyTorch refuses binary cross-entropy on probabilities under CUDA bf16 autocast, which the MCLoss needs
because of the max-constraint. The CPU tests could not show this. Fix: the MCLoss is computed in
float32 with autocast off; a test checks it equals the CPU loss. A `preflight` command runs one bf16
forward/backward/inference step for all 37 arm variants before a run resumes. Only B6 was affected and
no B6 run had finished. Tagged `v2-frozen-1`. **No test result seen.**

## Deviation 2 — 13 Oct 2026 · Steps 6-7 · phase schedules brought in line with the protocol text

Found during the repository rebuild (`hlg-net-v2`), before any test fold was opened:

1. **B4 (cRT) phase 1** retrained a plain-CE model inside each B4 candidate with the default learning
   rate instead of using "the fold's selected B0". Now the runner trains B0 first and B4 loads the
   selected B0 checkpoint (`init_from`, recorded in `meta.json`); only phase 2 (frozen backbone, new
   head, q = 1 sampler, 15 epochs) is trained, and the checkpoint is selected within phase 2 only.
2. **B3 (LDAM-DRW)** could early-stop before the DRW phase (80 % of the cap) and select a checkpoint
   from phase 1. Now B3 runs the fixed 60-epoch budget; DRW starts at epoch 48; the checkpoint is
   selected within the DRW phase only.
3. **M-prob** now uses the fixed epoch budget (no early stopping), as the protocol's Rules state.
4. **Arms derived from a B4 B\*** keep the gate and the boost frozen in phase 2 ("add the gate in
   phase 1 and keep it frozen in phase 2"); before, phase 2 also trained the gate.
5. **B6 on a loss-based B\*** (B2 or B3) now uses the B1 sampler (q = 0.5) with the MCLoss, as the
   Step 6 table states; before, the MCLoss simply replaced the objective and the rebalancing was lost.

Consequence: the runs finished under `v2-frozen-1` on fold r1/k0 (Tier 1 and B5) were produced by
code that differs for B3, B4 and the derived arms, and they inherit B\* from that selection. They are
discarded and the fold is rerun from scratch under the new tag. **No test result seen** (no test
prediction file had been opened; the primary analysis had not been run).

## Deviation 3 — 13 Oct 2026 · Step 7 · augmentation random streams

The training transform drew every augmentation from one NumPy generator that was copied into each
DataLoader worker at fork, so workers replayed the same parameter sequence on different images.
Now each sample draws its own generator seeded from (run seed, the worker's torch seed, sample index,
call counter): streams differ across workers and epochs and are reproducible given the run seed
(tested). Same consequence as Deviation 2: fold r1/k0 is rerun. **No test result seen.**

## Deviation 4 — 13 Oct 2026 · Steps 1, 7, 9, 11 · provenance and bookkeeping (no numerical effect)

- `meta.json` now carries every Step 1 field (git commit + dirty flag, config hash, manifest and split
  SHA-256, GPU model and driver, start/end time, best epoch and inner-val score, selection record,
  pretrained-weight SHA-256, timm version, phase log).
- Prediction files carry `image_group`, `split_sha`, `manifest_sha`, `commit`, and (B5) the species
  head output; `species macro-F1` is now computable.
- Every run asserts that the split files and the manifest match `splits/<unit>/digests.txt` before
  reading any cell (Step 7, run procedure 1).
- The selection log is idempotent on resume and records the backbone, run id, divergence flag and
  rerun count; the divergence rule of Step 10 (rerun once on non-finite loss or inner-val score
  below B0 − 0.10) is implemented.
- `runs/<unit>/queue.csv` is written (tier, repeat, fold, arm order randomised with seed 7).

## Deviation 5 — 13 Oct 2026 · Step 2 / gate G1 · label provenance check tightened

`malprep gate` now fails while `label_source` in `configs/data.yaml` still contains a placeholder
(`<fill in>`). The frozen data of 12 Oct 2026 was accepted with the placeholder present. Fill in the
source of the polygon label files, then run `python -m malprep freeze --amend --reason "label_source
recorded"`; the amendment is stored in `FROZEN.json`. No data file changes.

## Open advisory items (carried over from the freeze)

- Magnification-matched BBBC041 crops not written (`bbbc041.mpidb_rbc_diameter_px` unset).
- NIH patient split generated rather than imported from the transfer study.
- GPU runs with memory-efficient attention are seed-fixed but not bit-reproducible (PyTorch); the
  Step 1 determinism check (`python -m malexp determinism`) records the measured spread.
