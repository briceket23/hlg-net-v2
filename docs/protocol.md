# HLG-Net confirmatory study: step-by-step experimental protocol

Oct 11, 2026 · @Tchinda Brice

## 0. Purpose and fixed decisions

This is the operating protocol for the confirmatory HLG-Net study. It turns the design in the companion protocol doc (section "Recommended optimal design") into numbered steps.

**Revision 2 (after the external methodological audit).** The changes are:

- All tuning and the choice of B\* are nested inside each outer training part. The global tuning split is removed.
- Each arm gets 2 configs per fold. The gate weight grid includes the earlier value 0.15.
- The boost starts near zero.
- The gate BCE weight comes from the sampled prevalence.
- The external macro-F1 always covers four stages.
- Folds are reduced, never stages merged.
- A BBBC041 exposure audit and a GPU pilot are added.
- Compute is re-estimated without NIH.
- Scope is stated as a stage-only study (audit Path B). It does not test the full multi-task HLG-Net.

Each step has three parts:

- **Do:** the actions.
- **Output:** the files the step must produce.
- **Check:** a pass/fail acceptance test. Do not start the next step until the check passes.

Record any change to a step in the deviations log (Step 16) *before* running it.

**Decisions fixed by this protocol:**

| Item | Fixed value |
| --- | --- |
| Primary question | Stage-only scope: does a supervised rare-stage gate with a bounded logit boost (M-full) improve four-class stage macro-F1 on annotated parasite crops over the best matched rebalanced baseline B\*? |
| Training/test data | MP-IDB, four species, stage labels ring / trophozoite / schizont / gametocyte |
| External data | BBBC041 parasitised cells with stage labels (zero-shot, no tuning) |
| Not used | NIH (no stage labels) |
| Grouping unit | Field image |
| Outer protocol | 3 repeats × 5 folds, image-grouped, rare-bucket stratified |
| Selection data | Nested: every hyperparameter, B\* and checkpoint is chosen on the inner validation split of each outer training part; no global tuning split |
| Backbone | MobileViT-XXS (ImageNet-1k pretrained); EfficientNet-B0 replication for B\* vs M-full |
| Primary endpoint | Per-fold four-class stage macro-F1 |
| Primary test | Corrected resampled t-test (Nadeau & Bengio 2003) on 15 paired fold differences |
| Decision rule | Claim H1 only if the 95 % CI lower bound > 0 **and** Δ ≥ 0.02 |
| Test-fold use | Inference once per trained model; never used for any choice |

**Roles.** If one person runs everything, the "analyst" steps (13–14) must still be scripted and run only after all runs in the matrix are finished. Do not inspect partial test results.

## Step 1. Environment, repository and provenance

Every result file must be traceable to one commit, one config, one data manifest and one split manifest.

**Do:**

1. Create a new repository `hlgnet-confirmatory/`. Copy only the modules you need from `src/`, then apply the Step 8 changes. Tag the starting commit `v0-import`.
2. Pin the environment in `environment.yml` or `requirements.lock`:
   - Python;
   - PyTorch and CUDA;
   - timm (MobileViT-XXS and EfficientNet-B0 weights);
   - scikit-learn, numpy, pandas, scipy, statsmodels, matplotlib.

   Export it with `pip freeze > env/freeze.txt`.
3. Fix the folder layout:

```
data/raw/            # untouched downloads (read-only)
data/manifests/      # cells.csv, bbbc041_cells.csv, digests
data/crops/          # generated crops (reproducible from manifest)
splits/              # outer_r{1..3}.json, digests
configs/arms/        # one YAML per arm, frozen after Step 9
runs/{arm}/{backbone}/r{r}_f{k}/   # checkpoint, log, preds, meta.json
analysis/            # scripts + generated tables/figures
logs/deviations.md   # Step 16
```

4. Set determinism in a single `seed_everything(seed)`:
   - seed Python, NumPy and torch;
   - `torch.backends.cudnn.benchmark=False`;
   - `torch.use_deterministic_algorithms(True, warn_only=True)`;
   - seed each DataLoader worker through a `generator` and `worker_init_fn`.
5. Write a `meta.json` in every run folder with:
   - git commit hash and dirty flag;
   - config hash;
   - data manifest SHA-256 and split manifest SHA-256;
   - arm, backbone, repeat, fold, init seed;
   - GPU model and driver;
   - start and end time;
   - best epoch;
   - inner-validation score at the best epoch.
6. Run every job on the same GPU model, or record the GPU model and keep each paired comparison on one machine.

**Output:** pinned environment files, repository skeleton, `seed_everything`, `write_meta`.

**Check:**

- Two runs of the same config and seed on the same machine give inner-validation macro-F1 within 0.005.
- `meta.json` contains every field listed above.

If the first check fails, record the measured nondeterminism in the deviations log. It becomes part of the noise floor.

## Steps 2–3. MP-IDB manifest, crops and QC

The manifest is built once and then frozen. Every later step reads cells only through it.

### Step 2. Acquisition and cell manifest

**Do:**

1. Download MP-IDB from the authors' official repository into `data/raw/mpidb/`. Record the URL, the commit or release, and the download date. Compute SHA-256 for every file into `data/raw/mpidb.sha256`.
2. Choose the source of per-cell labels and record it. Use the per-parasite annotation (bounding box + stage), if your copy has one.
   - Do **not** label connected components of the GT mask from the image-file-name stage codes when an image contains more than one stage. Those codes list the stages present in the image, not the stage of each cell.
   - If no per-cell file exists, label multi-stage images by hand (two annotators on rare stages) or exclude them, and report which.
3. Write `data/manifests/cells.csv`, one row per parasite:
   - `cell_id`, `image_id`, `species`, `stage`;
   - `x0,y0,x1,y1` (original pixels), `mask_area_px`;
   - `touches_border`, `label_source`.
4. Check for duplicate images:
   - Compute a perceptual hash of every field image.
   - Images with Hamming distance ≤ 4 are near-duplicates. Merge them into one `image_group`. Grouping uses `image_group`, not `image_id`.
5. Fix the exclusion rules in advance:
   - `mask_area_px` below a minimum (e.g. 1st percentile or < 100 px);
   - unknown or ambiguous stage;
   - boxes that contain two touching parasites of different stages.

   Log every exclusion with a reason code.

**Output:** `cells.csv`, `exclusions.csv`, `mpidb.sha256`, `data/manifests/cells.sha256`.

### Step 3. Crop extraction and QC

**Do:**

1. Crop rule (one rule for all cells):
   - Square box centred on the parasite box, with side = 1.5 × max(width, height). Clip to a minimum of 64 px.
   - Pad outside the image with the per-image median background colour.
   - Resize to 256 × 256 (bicubic) for MobileViT and 224 × 224 for EfficientNet-B0. Do the resize at load time from one stored crop size.
2. Save crops as lossless PNG at `data/crops/{cell_id}.png`. Keep the crop box in `cells.csv`.
3. No colour normalisation is baked into stored crops. Stain variation is handled by augmentation (Step 7).
4. Visual QC:
   - Inspect **every** schizont and gametocyte crop, plus a random 100 of ring and trophozoite.
   - Mark mislabels, double parasites and off-centre crops in `qc.csv`.
   - Fix or exclude, then re-freeze.
5. Write **Table 1** of the paper from the frozen manifest. For each species, give the number of images and image groups, and the cells per stage. Add the number of images that contain each rare stage. This number bounds how many folds can contain that stage.

**Output:** `data/crops/`, `qc.csv`, `analysis/table1_data.csv`, frozen `cells.sha256`.

**Check:**

- Re-running crop extraction from the manifest reproduces byte-identical PNGs.
- Every rare-stage crop has been inspected.
- The Table 1 totals equal the number of manifest rows minus exclusions.
- **Feasibility stop rule:** if a rare stage occurs in fewer than 5 image groups, 5-fold coverage of that stage is impossible. Then use 3 outer folds. If 3 folds still cannot place each rare stage in every test fold and inner validation set, stop: the data cannot support a four-class confirmatory estimate. Never merge stages while reporting a four-stage endpoint.

### Step 2 result: audit of the per-parasite label files (11 Oct 2026)

The uploaded polygon labels (YOLO format, 16 species × stage classes) provide a per-parasite stage label for every object. So the labelling question of Step 2 is answered. They do **not** match the dataset the manuscript describes.

| Check | Result |
| --- | --- |
| Files / instances | 209 label files (MP-IDB has 210 images; one ovale file is missing: 28 instead of 29), 1,436 parasites |
| Stage counts (labels) | Ring 1,284 · Trophozoite 81 · Schizont 41 · Gametocyte 30 |
| Stage counts (manuscript) | Ring 845 · Trophozoite 385 · Schizont 41 · Gametocyte 50 (total 1,321). Only the schizont count agrees |
| Species per image | Always one species per image |
| Filename stage codes vs labels | Agree in all 209 files; 30 images contain more than one stage, and each of their objects is labelled individually |
| Filename anomaly | `1701151546-0008-R-T` uses a hyphen instead of an underscore; parse the prefix by the first hyphen |
| Candidate slide groups (filename prefix) | 17 prefixes, each of a single species, 1–25 images each |
| Rare-stage support | Schizonts in 38 images / 10 prefixes; gametocytes in 26 images / 7 prefixes |
| Ring-stage concentration | 1,230 of 1,284 rings are *P. falciparum*; three prefixes hold 1,061 cells |
| Object size (px, assuming 2592 × 1944 images) | Median box side: ring 64, trophozoite 87, schizont 119, gametocyte 167; max 223 |
| Quality | No empty files, no duplicate or overlapping objects, none touching the border, 1 object under 100 px² |

**Why the counts differ:** the HLG-Net loader reads pre-cropped images from `MPIDB_DIR/species/stage/` folders. Those crops carry no source-image ID, and their counts differ from these labels. They come from a different, undocumented source and must not be used for the confirmatory study.

**Decisions this forces:**

1. **Data source.** Build every crop from these polygon labels and the original MP-IDB images. Record where the label files came from (official release, Roboflow export, own annotation) before G1.
2. **Grouping unit.** The filename prefix very likely identifies the slide or smear. Confirm it from the MP-IDB documentation.
   - If confirmed, group by prefix (17 groups) for the primary analysis and use image-level grouping as a sensitivity analysis. The difference between the two is itself evidence for the thesis split-protocol chapter (RQ1).
   - Simulated rare-bucket 5-fold assignment by prefix places ≥ 5 gametocytes and ≥ 6 schizonts in every test fold. However, test-fold sizes range from 85 to 807 cells, because one prefix alone holds 416 cells.
   - By image, 5 folds are balanced (≈ 287 cells each), with ≥ 6 gametocytes and ≥ 8 schizonts per test fold.
   - With only 17 slide groups, repeats give few distinct partitions. The assignment must randomise tie-breaking, and the protocol reports how many distinct partitions the 3 repeats produced.
3. **Crop rule (replaces the Step 3 resize rule).** Cut a fixed 256 × 256 window at native resolution, centred on each object, with no resizing.
   - It fits the largest object (223 px).
   - It matches the MobileViT pretrained input size.
   - It keeps absolute size, which is a real morphological cue between stages.
   - Resizing each box to a fixed size would remove that cue.
4. **Confound to report.** Stage is entangled with species and slide: rings come almost only from three high-parasitaemia falciparum slides, and gametocytes mostly from non-falciparum slides. Report per-species stage confusion matrices and the slide-grouped result. Do not interpret stage accuracy as species-independent.
5. **Thesis and manuscript counts.** Replace Table 2 and every ratio (93.1 %, 1,253:1, the NIH batch share) with counts generated from this manifest. Ring + trophozoite share = 1,365 / 1,436 = 95.1 %.

Files: `mpidb_instance_manifest_draft.csv` (one row per parasite, with prefix group) and `audit_mpidb_labels.py` (reproduces this table).

**Implementation.** Steps 2–5 are implemented in the `malprep` package: manifest, duplicate grouping, crops, QC, Table 1, BBBC041, splits and gate checks. `python -m malprep gate` reports gates G1 and G2. A dry run on the real labels gave 1,435 parasites after one AREA exclusion; slide grouping gave 17 groups, and every fold passed the leakage and coverage tests.

Two rules were added during implementation:

- A near-duplicate needs a thumbnail correlation ≥ 0.95 as well as a pHash match. pHash alone flagged 143 false pairs of smear fields.
- The inner training set must contain all four stages.

**Verification round (12 Oct 2026).** The run on the real data passed every gate check. A code and data review then found and fixed four problems:

1. **Inner validation size.** Under slide grouping, inner validation took 4 %–58 % of the training part. It is now chosen as the best of 100 candidates, and in a dry run on the real labels took 12 %–24 %.
2. **Test-fold size.** Test folds ranged from 82 to 807 cells; they now range from 101 to 456. The largest slide alone holds 416 cells.
3. **Repeated test folds.** Balancing made the repeats converge on the same partition. A penalty now limits this to 2 of 15 slide-level test folds.
4. **Portability.** Stored paths now use forward slashes and files use LF line endings, so manifests made on Windows work on Linux servers. A `freeze` lock now blocks any later rewrite.

**Statistical consequence for Step 14.** With only 17 slides, the three slide-level repeats are not independent resplits. Report the number of distinct test folds (13 of 15 in the dry run) next to the corrected t-test, and state it as a limitation. The image-level sensitivity analysis has no repeated folds.

**Gates G1 and G2 passed: data frozen on 12 Oct 2026 at 17:35 (FROZEN.json, 34 files).** An independent check of the uploaded manifests found all 11 hashes matching. The frozen set holds 1,435 MP-IDB parasites (R 1,283 / T 81 / S 41 / G 30) on 17 slides, with all 71 rare crops reviewed and none excluded or relabelled; 2,452 BBBC041 stage cells; and 27,558 NIH cells in 200 patient groups. Paths are portable. Two advisory items stay open and are recorded as deviations: the magnification-matched BBBC041 crops were not written (`mpidb_rbc_diameter_px` unset), and the NIH patient split was generated rather than imported from the transfer study. The provenance records `commit: no-git`, so the frozen files must now be committed and tagged.

## Step 4. BBBC041 external stage set

BBBC041 is used only for zero-shot testing of frozen models. It never influences training, tuning, thresholds or temperature.

**Do:**

1. Download BBBC041 (P. vivax thin smears) from the Broad Bioimage Benchmark Collection into `data/raw/bbbc041/`. Record the URL, the date and SHA-256 of every file.
2. Parse the training and test annotation files.
   - Keep objects labelled `ring`, `trophozoite`, `schizont` and `gametocyte`.
   - Drop `difficult`, `leukocyte` and `red blood cell` (the last is kept only if you run the optional negatives analysis).
   - Verify the category names and counts in your copy before relying on them.
3. Write `data/manifests/bbbc041_cells.csv` with the same columns as `cells.csv`. Set `species=vivax`. Add the `subset` column (official train/test) and an image-format column.
4. Extract crops with the **same crop rule as Step 3**:
   - The box comes from the annotation, so derive a mask-free size from the box.
   - BBBC041 images have a different magnification and resolution. Record the median parasite box side in pixels for both datasets.
   - Report the ratio. Do not rescale to match, because the primary external analysis is "as acquired". A rescaled sensitivity analysis is optional and must be declared here.
5. Run QC on 50 random crops per stage.
6. Write **Table 5a**: cells per stage, number of images, and the official train/test split.

**Exposure audit (before calling it zero-shot).** Confirm that no BBBC041 image entered any pretraining, fine-tuning, model selection or crop-rule decision for the models in this study. The models start from ImageNet weights and train on MP-IDB only. BBBC041 served as an external set in the transfer study, which is a separate study. Record this in a short exposure statement. If any exposure is found, call the result "external transfer after exposure", not zero-shot.

**Checked on the BBBC041 annotation files (12 Oct 2026).**

| Subset | Images | Format and size | Red-cell box (median) | Ring | Trophozoite | Schizont | Gametocyte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training.json | 1,208 | PNG 1600 × 1200 | 113 px | 353 | 1,473 | 179 | 144 |
| test.json | 120 | JPG 1944 × 1383 | 144 px | 169 | 111 | 11 | 12 |

Consequences for the protocol:

1. **Geometry.** BBBC041 boxes enclose the whole infected red cell, while MP-IDB polygons outline the parasite. Magnification is therefore compared through red-cell diameter, not object size.
   - `malprep rbc-scale` estimates the MP-IDB red-cell diameter.
   - The as-acquired crops are primary. Magnification-matched crops are a declared sensitivity analysis.
2. **Within-dataset shift.** The two subsets differ in magnification (×1.27), format and image size. Always report them separately.
3. **External endpoint.** Pool both subsets, since neither enters this study, and report each subset separately. The official test subset alone has only 11 schizonts and 12 gametocytes.
4. **Prior shift.** BBBC041 is 66 % trophozoites, while MP-IDB is 89 % rings. Report per-stage recall and the confusion matrix next to macro-F1, and do not re-estimate priors on BBBC041.

**Output:** `bbbc041_cells.csv`, crops in `data/crops_bbbc041/`, `bbbc041_cells.sha256`, `table4a_external_data.csv`.

**Check:**

- The stage counts equal the annotation counts minus the documented exclusions.
- The magnification ratio is recorded.
- The external macro-F1 always covers all four stages, so it is the same endpoint as on MP-IDB. A stage with fewer than 30 cells is flagged as low-support, and its per-stage F1 is shown with its interval.

## Step 5. Splits: tuning split, outer folds and inner validation

All splits are generated once, before any model is trained. They are hashed and never regenerated.

**Seeds:**

- Split seeds: outer repeats r1, r2, r3 = 101, 102, 103.
- Initialisation seed for run (r, k) = 1000·r + k.

Keeping split and initialisation seeds separate lets a repeat vary both the partition and the initialisation.

**Do:**

1. Implement the **image-grouped rare-bucket assignment** `assign_groups(groups, K, seed)`:
   1. For each `image_group`, count its cells per stage and record its species.
   2. Bucket the groups:
      - **G**: groups containing any gametocyte;
      - **S**: groups containing a schizont and no gametocyte;
      - **C**: all other groups.
   3. Shuffle within each bucket with `seed`.
   4. Allocate bucket G, then S, then C. Put each group into the fold with the fewest cells of that bucket's stage so far. Break ties by fewest total cells, then by the lowest count of that group's species.
   5. Return `group → fold`.
2. **Outer folds:** for r in {1, 2, 3}, run `assign_groups(all groups, K=5, seed=100+r)`. Save `splits/outer_r{r}.json`.
3. **Inner validation (the only selection data):** for each outer training part, run `assign_groups(train groups, K=5, seed=10·(100+r)+k)` and take fold 0 as inner validation (≈ 20 % of training groups). The rest is the inner training set. Save it in the same JSON under `inner_val`.
   - Every choice is made on this split: early stopping, temperature, the arm's config, B\* and the gate weight.
4. **No global tuning split.** Outer-test groups never take part in any tuning, checkpoint or selection decision for their own fold.
   - This is ordinary nested cross-validation.
   - Shared settings fixed before any training (resolution, augmentation, epoch cap, weight decay) are declared a priori and never tuned.
5. Write the per-fold coverage table, one per repeat. For each stage, give cells and image groups per fold, plus the species mix per fold. This table fills thesis Table 4.x (tab:int-coverage).

**Output:** `splits/outer_r{1,2,3}.json` (test, inner\_val and inner\_train group lists per fold), `splits/*.sha256`, `analysis/table_fold_coverage.csv`.

**Check (automated test `test_splits.py`; all must pass):**

- In every (r, k), the inner\_train, inner\_val and test groups are pairwise disjoint, and their union equals all groups.
- Within each repeat, every group is in exactly one test fold.
- No `image_id` and no near-duplicate group crosses partitions.
- Every outer test fold contains ≥ 1 schizont group and ≥ 1 gametocyte group. If impossible, apply the Step 3 feasibility rule.
- Every inner validation set contains ≥ 1 group of each rare stage. If it does not, rerun that inner assignment with seed + 1 and log it.
- The species share per test fold is within ± 10 percentage points of the overall share. This is reported, not enforced.
- The split SHA-256 digests are written into `splits/digests.txt` and referenced by every `meta.json`.

## Step 6. Model and arm definitions

Every arm uses one network:

- a backbone producing pooled features z ∈ R^d;
- a linear stage head producing logits s ∈ R^4.

Stages are indexed ring = 0, trophozoite = 1, schizont = 2, gametocyte = 3. The rare-stage group is L = {2, 3} (schizont, gametocyte). This grouping serves the classifier only; it is not a claim that gametocytes follow schizonts in development. A common boost on both logits changes rare-vs-common odds; it cannot separate schizont from gametocyte. n\_c is the training count of stage c and π\_c = n\_c / Σn is its prior; both are computed on the inner training set of each fold.

**Removed from every arm:**

- the p\_uninfected / detection suppression term;
- the NIH data and detection head;
- the projection head and SupCon loss;
- PCGrad;
- the prototype memory bank;
- the minority injector;
- stage mixup;
- the species-weighted sampler;
- TTA.

The last five can come back only as Step 15 ablations.

### Arms

| Arm | Training objective | Sampler | Tuned hyperparameters (grid) |
| --- | --- | --- | --- |
| B0 | CE(s, y) | instance (uniform over cells) | lr |
| B1 | CE(s, y) | weight per cell ∝ n\_y^(−q) | lr; q = 0.5 fixed |
| B2 | logit-adjusted CE, below | instance | lr; τ = 1 fixed (Balanced Softmax) |
| B3 | LDAM + deferred re-weighting, below | instance | lr; max margin 0.5, s = 30, β = 0.9999 fixed |
| B4 | cRT: phase 1 = B0; phase 2 = backbone frozen, new linear head trained 15 epochs with sampler q = 1 | phase-specific | phase 1 = the fold's selected B0; head lr ∈ {1e-3, 1e-2} |
| B5 | B\* objective + λ\_sp · CE(species head, species) | B\*'s | λ\_sp ∈ {0.3, 1.0} |
| B6 | B\* rebalancing + C-HMCNN constraint module and MCLoss on the hierarchy {early: ring, troph; late: schizont, gametocyte} | B\*'s; if B\* is loss-based, the B1 sampler with B1's q | lr |
| M-aux | B\* objective + λ\_g · BCE(g, 1\[y ∈ L\]) | B\*'s | λ\_g ∈ {0.15, 0.5}; lr = the fold's B\* lr |
| M-full | B\* objective on boosted logits s′ + λ\_g · BCE(g, 1\[y ∈ L\]) | B\*'s | λ\_g ∈ {0.15, 0.5}; lr = the fold's B\* lr |
| M-prob (optional) | NLL of the composed probability, below | B\*'s | λ\_prior ∈ {0, 1}; lr = the fold's B\* lr |

The shared learning-rate grid is lr ∈ {3e-4, 1e-3}. Every arm gets exactly 2 configs per outer fold, so tuning budgets are equal and nested tuning stays affordable (Step 9). The gate grid contains 0.15, the value selected in the exploratory sweep, and one stronger value.

### Formulas

Logit-adjusted CE (B2), with prediction = argmax s:

```latex
\mathcal{L}_{LA} = -\log \frac{\exp(s_y + \tau\log\pi_y)}{\sum_{c}\exp(s_c + \tau\log\pi_c)}
```

LDAM-DRW (B3). Features and head weights are L2-normalised and scaled by s = 30. From 80 % of the fixed epoch budget onward, the loss is re-weighted by w\_c = (1 − β) / (1 − β^(n\_c)):

```latex
\mathcal{L}_{LDAM} = -\log \frac{e^{s(\cos\theta_y - \Delta_y)}}{e^{s(\cos\theta_y - \Delta_y)} + \sum_{c\neq y} e^{s\cos\theta_c}},\qquad \Delta_c = \frac{C}{n_c^{1/4}}
```

Gate (M-aux, M-full). h is an MLP d→64→1 with GELU and dropout 0.1. The input z is **not detached**, so the gate loss shapes the backbone. The BCE pos\_weight is computed from the prevalence of L as sampled in training batches (measured over one epoch of the arm's sampler), not from raw counts, so a rebalancing sampler is not counted twice. Log the gate's mean output per class every epoch.

```latex
g = \sigma\big(h(z)\big),\qquad \mathcal{L}_{gate} = \mathrm{BCE}\big(g,\ \mathbb{1}[y\in L]\big)
```

Boost (M-full only). The boost is applied **before** any logit adjustment. Prediction = argmax s′.

```latex
s'_c = s_c + b\,g\ \ (c\in L),\qquad s'_c = s_c\ \ (c\notin L),\qquad b = 5\,\sigma(\beta),\ \beta_0 = -2.94\ (b_0 \approx 0.25)
```

M-full total loss:

```latex
\mathcal{L} = \mathcal{L}_{B^*}(s', y) + \lambda_g\,\mathcal{L}_{gate}
```

M-prob, with group(c) being early or late. Prediction = argmax log p:

```latex
\log p(c\mid x) = \log p\big(\mathrm{group}(c)\mid x\big) + \log \mathrm{softmax}_{c' \in \mathrm{group}(c)}(s)_c,\quad p(L\mid x)=g
```

### Rules

- Only B3, B4 and M-prob have phase-dependent schedules. They use a fixed epoch budget, and the checkpoint is selected on inner validation **within the final phase only**.
- B\* is chosen per outer fold among B1–B4 on inner validation (Step 9). It is reported as "the best selected rebalanced baseline", and B0 stays the plain-CE anchor. If B\* is B4 (cRT), M-aux and M-full add the gate in phase 1 and keep it frozen in phase 2.
- Every flag lives in one YAML per arm. Each component has one boolean (`use_gate`, `use_boost`, `detach_gate`, `use_species_head`, `use_chmcnn`, ...).

## Step 7. Shared training procedure (one run)

Every run of every arm follows this procedure. Only the arm YAML changes.

| Setting | Value |
| --- | --- |
| Backbone | `mobilevit_xxs` (timm mobilevit\_xxs.cvnets\_in1k, model-card preprocessing; log the timm version and weight SHA-256), input 256 fixed a priori (the manuscript's 128 px only as a declared sensitivity run); replication: `efficientnet_b0`, input 224 |
| Head init | Linear, zero bias; gate MLP as Step 6 |
| Optimiser | AdamW, weight decay 0.05 (none on bias and norm parameters; a declared change from the manuscript's 1e-4: every arm is rerun under it, so it is not a confound, but February numbers are not comparable), lr from the arm grid, head lr = 10 × backbone lr |
| Schedule | 5-epoch linear warmup, then cosine decay to 1 % of lr |
| Epoch budget | 60 epochs fixed cap. B3, B4 and M-prob use the fixed phase budgets of Step 6 |
| Batch | 64 if the Step 8 GPU pilot fits it, else 32; the same physical batch for every arm (no contrastive loss, so no accumulation is needed); gradient clipping at norm 1.0; bf16 autocast once the smoke test passes |
| Augmentation (train only) | Random horizontal and vertical flip; random 90° rotation; RandomResizedCrop scale (0.85, 1.0), ratio (0.9, 1.1); HED stain jitter σ = 0.03; brightness/contrast ± 0.1; ImageNet normalisation |
| Evaluation transform | Resize only + normalisation; no TTA |
| Model selection | After each epoch, compute stage macro-F1 on **inner validation**. Keep the best checkpoint. Stop when it has not improved by 0.002 for 15 epochs |
| Calibration | After training, fit one temperature T on inner-validation logits (L-BFGS, NLL). Store T; it changes probabilities, not argmax |
| Logged per epoch | Train loss components, inner-val macro-F1 and per-class F1, lr, b (M-full), mean gate output by class |

**Run procedure:**

1. Load the arm YAML, the split file and the fold. Assert that the split and manifest digests match `splits/digests.txt`.
2. Build the datasets: inner\_train, inner\_val and test. Compute π and the sampler weights from **inner\_train only**.
3. Run `seed_everything(1000·r + k)`. Build the model and the optimiser.
4. Train with early stopping on inner validation. **The test DataLoader is not constructed during training.** This is enforced in code, with a test that fails if test indices are touched before `finalize()`.
5. `finalize()`:
   1. Load the best checkpoint.
   2. Fit T on inner validation.
   3. Construct the test loader and run inference once on the outer test fold.
   4. If Step 12 is enabled, also run inference on BBBC041.
   5. Write the prediction files (Step 11) and `meta.json`.
6. Delete optimiser states and keep the best weights only (≈ 5 MB for MobileViT-XXS).

**Check:**

- The run's `meta.json` shows best epoch < cap, or a flag is set when it equals the cap.
- The prediction file has exactly one row per test cell.
- No test metric appears in the training log.

## Steps 8–9. Code verification, tuning and choice of B\*

### Step 8. Code verification (pytest, before any real run)

| Test | Pass condition |
| --- | --- |
| `test_splits.py` | All Step 5 checks |
| `test_no_test_access` | Training raises an error if any test index is loaded before `finalize()` |
| `test_metrics` | Macro-F1, per-class F1, balanced accuracy and confusion matrix equal sklearn's (`labels=[0,1,2,3]`, `zero_division=0`) on 100 random prediction sets |
| `test_ece` | ECE (15 equal-width bins, top-label) equals a reference implementation to 1e-6 |
| `test_flags` | For each flag, toggling it changes only the intended parameters and loss terms (compare parameter-name sets and the loss dictionary keys) |
| `test_boost_off` | With `use_boost=False`, s′ = s exactly; with b forced to 0, M-full logits equal M-aux logits |
| `test_gate_grad` | Gate loss alone gives non-zero backbone gradients when `detach_gate=False` and exactly zero when `detach_gate=True` |
| `test_logit_adjust` | τ = 0 reproduces plain CE |
| `test_ldam` | Zero margins reproduce scaled cosine CE |
| `test_sampler` | Empirical class frequencies over 20 000 draws within ± 2 points of the target |
| `test_shift_invariance` | Adding the same constant to all logits leaves CE and argmax unchanged (documents why the suppression term was removed) |
| `test_overfit` | 64 training cells reach training accuracy > 0.95 in 100 steps for every arm |
| `smoke` | 2 epochs on outer fold (r1, k0) for every arm; `finalize()` writes valid prediction and `meta.json` files |
| GPU pilot (on the actual machine) | One full 60-epoch run of M-full on the inner split of fold (r1, k0). Record peak memory, epoch time, GPU model, VRAM and driver. Fix the physical batch and re-estimate compute from the measured time. The pilot chooses nothing except the batch size, and its scores are never reported. |

**Check:** all tests pass in CI on the tagged commit. Tag it `v1-code-verified`.

**Implementation (12 Oct 2026).** Steps 6–11 are implemented in the `malexp` package: arm configs with one flag per component, the network (backbone, stage head, species head, gate, boost, M-prob composition, Coordinate Attention), the losses (CE, logit-adjusted, LDAM-DRW, C-HMCNN MCLoss, gate BCE), the shared training recipe with the held-out guard, temperature scaling, prediction files, nested selection and a resumable fold runner. The 16 Step-8 tests pass on CPU, including the nested smoke run of all 13 arms. The GPU pilot, the `v2-frozen` tag and the analysis script (Steps 13–14) remain to be done.

**GPU pilot result (13 Oct 2026).** M-full, inner split of (r1, k0), MobileViT-XXS 256 px, batch 64, bf16: NVIDIA GeForce RTX 4050 Laptop GPU, torch 2.6.0+cu124, 4.7 s per epoch, peak allocated GPU memory 2,095 MB, 19 epochs before early stopping (89.6 s). Decision: physical batch 64 for every arm. One CUDA-only defect was found and fixed by the pilot (a broadcasting index assignment in the boost, replaced by a mask product; identical numerics, tests unchanged). PyTorch reports that the memory-efficient attention backward is non-deterministic on CUDA, so GPU runs are seed-fixed but not bit-reproducible; this is recorded in the limitations. The pilot score is not reported. The analysis script (`analysis/run_analysis.py`, Steps 13–14) is implemented and tested on a synthetic 180-run tree; the `v2-frozen` tag remains to be done.

**Deviation 1 (13 Oct 2026, during fold r1/k0 of the slide-grouped matrix).** The first B6 candidate stopped at its first training step on the GPU. Under CUDA bf16 autocast PyTorch refuses binary cross-entropy on probabilities, which the C-HMCNN MCLoss needs because of the max-constraint. The CPU tests could not show this, because CPU autocast allows the operation. Fix: the MCLoss is computed in float32 with autocast switched off. On the same inputs it gives the same value as the CPU loss, which a new test checks. Only B6 is affected, and no B6 run had finished. Tier-1 runs and the finished B5 runs were not retrained, and the code paths they use are unchanged. A `preflight` command was added: it runs one bf16 forward, backward and inference step on the GPU for each of the 37 arm variants the matrix can produce, before the run resumes. The fix is committed as `v2-frozen-1` on top of `v2-frozen`.

### Step 9. Nested tuning and per-fold selection of B\*

All selection happens inside each outer training part, on its inner validation split. There is no global tuning split, and test predictions are produced only for the selected run of each arm in each fold.

1. **Per fold, per arm:**
   - Train the arm's 2 configs (Step 6) on inner\_train.
   - Select the one with the higher inner-validation macro-F1. If they are within 0.005, take the smaller lr or the smaller gate weight.
   - The selected run is the fold model. It is not retrained on inner\_train + inner\_val, because inner validation is still needed for stopping and temperature.
2. **Order within a fold:**
   - Run B0–B4 first.
   - B\*(r, k) = the best of B1–B4 on inner validation. If two are within 0.005, take the simpler one, in the order B2 < B1 < B4 < B3.
   - Then run B5, B6, M-aux, M-full and M-prob on that fold's B\*, inheriting its rebalancing and lr.
3. **Ablations (Step 15)** inherit the fold's selected M-full settings, with no extra tuning.
4. **EfficientNet-B0 replication:** per fold, use the same B\* family as MobileViT and tune only the 2-value lr grid. M-full is built on it as above.
5. **Selection log:** write `analysis/selection.csv` with one row per (r, k, arm, config). It holds the inner-val score, the selected flag, and B\*(r, k). Report how often each arm was chosen as B\*, and the gate weights selected.
6. **Freeze before any training:** grids, tie rules, epoch cap, weight decay, augmentation and resolution in `configs/arms/*.yaml` and `configs/digests.txt`. Tag the commit `v2-frozen`.

**Check:**

- `selection.csv` is complete.
- No test prediction file exists for a non-selected config.
- Every fold model's `meta.json` names its selection record.

The analysis plan (Steps 13–14), including the assumed σ\_d range for the power statement, is timestamped **before** Step 10. Upload it to OSF or commit it with a signed tag.

## Steps 10–11. Confirmatory runs and prediction files

### Step 10. Run matrix and execution order

Each cell of the matrix is one arm on one outer fold (r, k). It consists of 2 config runs, selected by nested tuning (Step 9). Ablations use 1 run because they inherit settings.

Per-run time, measured in the Step 8 GPU pilot: 4.7 s per epoch at batch 64 on an RTX 4050 Laptop GPU, i.e. at most 4.7 min for a run that reaches the 60-epoch cap and about 1.5–2.5 min for a typical early-stopped run (the pilot stopped at 19 epochs). The slide-grouped matrix (21 runs per fold × 15 folds = 315 runs; 345 with M-prob) therefore needs about 8–13 GPU-hours expected and 25 GPU-hours at worst; the image-grouped sensitivity analysis and the EfficientNet-B0 replication (Tiers 1–2, 270 runs) add a similar amount each. The earlier 0.2 GPU-h/run planning figure is superseded.

| Tier | Arms | Backbone | Runs | GPU-h (≈ 0.2 h/run) |
| --- | --- | --- | --- | --- |
| 1 (primary) | B\*, M-aux, M-full (B1–B4 must run first to pick B\*) | MobileViT-XXS | 90 | 18 |
| 2 (baseline ladder) | B0, the three non-selected arms of B1–B4, B5, B6 | MobileViT-XXS | 180 | 36 |
| 3 (ablations, Step 15) | M-full without aux loss; M-full with detached gate; M-full + Coordinate Attention | MobileViT-XXS | 45 | 9 |
| 4 (replication) | B\*, M-full | EfficientNet-B0 | 60 | 12 |
| 5 (optional) | M-prob | MobileViT-XXS | 30 | 6 |
| **Total** |  |  | **405** | **≈ 80** |

In practice, B0–B4 run first in every fold because B\* depends on them. Tiers 1 and 2 therefore finish together.

**Execution order:**

1. Generate `runs/queue.csv` from the matrix. Order it by tier, then repeat, then fold, and randomise the arm order within each (r, k) with seed 7.
   - Folds are completed for all arms of a tier before the next fold starts.
   - If compute runs out, every arm then has the same folds and the paired analysis stays valid.
2. A tier counts as complete only when all 15 folds exist for all its arms. The primary analysis needs Tier 1 complete; nothing else is required.
3. **Failures:**
   - A crashed run is rerun with the same seed and config, and the rerun is logged.
   - A run that finishes but diverges (training loss NaN, or inner-val macro-F1 below B0's tuning score − 0.10) is rerun **once** with the same seed. If it diverges again, it is kept and reported.
   - Never replace a run because its *test* score looks bad.
4. Do not open test predictions or compute any test metric until a tier is complete. Monitor only `meta.json` (best epoch, inner-val score).

### Step 11. Prediction files

Each run writes three Parquet files: `preds_test.parquet` (outer test fold), `preds_innerval.parquet` and `preds_bbbc041.parquet`. Each file has one row per cell and these columns:

| Column | Content |
| --- | --- |
| `cell_id`, `image_group`, `species`, `y_true` | From the manifest |
| `logit_0..3` | Raw stage logits as used for prediction (s′ for M-full) |
| `T`, `prob_0..3` | Temperature and calibrated probabilities |
| `y_pred` | argmax of the logits |
| `gate` | g (gate arms only; NaN otherwise) |
| `arm`, `backbone`, `repeat`, `fold`, `seed` | Run identifiers |
| `config_sha`, `split_sha`, `manifest_sha`, `commit` | Provenance |

**Check (script `check_completeness.py`):**

- Every (arm, backbone, r, k) in the matrix has all three files.
- Within each repeat, the union of `cell_id` in the test files equals the manifest, with no duplicates.
- All digests match the frozen ones.

Only then does Step 13 start.

## Step 12. External evaluation on BBBC041

The 15 frozen fold models of each arm are applied to BBBC041 without any change. The external result is a secondary endpoint and cannot rescue a failed primary.

**Do:**

1. For each run, compute predictions on all BBBC041 cells in `finalize()` with the same evaluation transform. Use the temperature T fitted on MP-IDB inner validation. No re-tuning, re-thresholding or BBBC041 temperature.
2. **Unit of replication:** the 15 fold models. Each model gives one external macro-F1, computed over the stages that pass the Step 4 count rule. The arm's external score is the mean over the 15 models, with a t-interval across models.
   - The 15 models share one external set and their training sets overlap. Treat the interval as descriptive, not as a test.
3. **Paired contrast:** M-full − B\* for each (r, k), reported with the mean and the range. Uncertainty from target sampling comes from an image-grouped bootstrap over BBBC041 images (2 000 resamples) applied to the fold-averaged probabilities of each arm.
4. **Report by subset:** official train vs official test images, and by image format if formats differ. This repeats the subset analysis of the transfer study.
5. **Species caveat:** MP-IDB contains P. vivax, but stage morphology and appearance differ by species and site. Report the per-stage confusion matrix so that readers can see whether errors are shifts between adjacent stages (ring↔trophozoite) or rare-stage misses.
6. **Optional sensitivity:** rescale BBBC041 crops to match the MP-IDB median parasite size (Step 4 ratio) and repeat item 2. This is declared here and reported as sensitivity only.

**Output:** `analysis/external_by_model.csv`, `analysis/external_bootstrap.csv`, `table4_external.tex`, an external confusion-matrix figure.

**Check:**

- No BBBC041 file is read by training or tuning code. Verify this with grep in CI and with the absence of a BBBC041 path in any training config.
- The external files have 15 models × the BBBC041 cell count rows per arm.

## Steps 13–14. Metrics, statistics and decision rule

One script, `analysis/run_analysis.py`, reads only the prediction files and writes every number in the paper. Freeze it (Step 9) before it sees test predictions.

### Step 13. Metrics

| Endpoint | Definition | Regime | Role |
| --- | --- | --- | --- |
| Stage macro-F1 | Unweighted mean of the four per-stage F1 values (`zero_division=0`) on the outer test fold | Per fold (15 values per arm) | **Primary** |
| Schizont F1, gametocyte F1 | Per-class F1 | **Pooled per repeat** (3 values per arm) + grouped bootstrap | Key secondary (H2) |
| Balanced accuracy | Mean per-stage recall | Per fold | Key secondary |
| External stage macro-F1 | Step 12 | Per model | Key secondary |
| Per-stage precision and recall, confusion matrix | — | Pooled per repeat | Descriptive |
| ECE (15 bins, top-label), Brier, NLL | On calibrated probabilities | Per fold | Secondary |
| Species macro-F1 | Only B5 | Per fold | Descriptive (species is image-level) |
| Params, MACs, CPU latency | Batch 1, 4 threads, 50 warm-up and 200 timed runs, median and IQR, one named CPU | Per arm | Descriptive |

Rare-stage F1 is pooled because a fold may hold only a few schizonts, which makes per-fold rare F1 values unstable. This choice is fixed here, not after seeing results.

**Pooled values** are labelled "pooled" in every table and never share a column with per-fold values.

### Step 14. Statistical analysis

**Primary contrast (H1).** For each (r, k), let d = macro-F1(M-full) − macro-F1(B\*). J = 15, and ρ = n\_test / n\_train ≈ 0.25 (cells, outer split). The corrected resampled t statistic (Nadeau & Bengio 2003; repeated k-fold form, Bouckaert & Frank 2004) has df = J − 1 = 14:

```latex
t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \rho\right)\hat\sigma_d^{2}}},\qquad \mathrm{CI}_{95} = \bar d \pm t_{0.975,\,14}\sqrt{\left(\frac{1}{J} + \rho\right)\hat\sigma_d^{2}}
```

**Decision rule (smallest effect of interest Δ\_min = 0.02):**

| Result | Wording in the paper |
| --- | --- |
| CI lower bound > 0 and d̄ ≥ 0.02 | M-full improves stage macro-F1 over B\* (H1 supported) |
| CI lower bound > 0 and d̄ < 0.02 | Detectable but below the smallest effect of interest |
| CI contains 0 and upper bound < 0.02 | No meaningful improvement |
| CI contains 0 and upper bound ≥ 0.02 | Inconclusive at this sample size |
| CI upper bound < 0 | M-full is worse than B\* |

**Secondary contrasts.** Use the same test for each, with Holm adjustment within each family:

- **Family A (H2), M-full vs B\*:**
  - schizont F1 and gametocyte F1: pooled per repeat, with grouped-bootstrap CIs on the repeat-averaged difference;
  - balanced accuracy;
  - external macro-F1, reported descriptively as in Step 12.
- **Family B, the ladder:** M-full vs each of B0–B6.
- **Family C (H3) and ablations:** M-full vs M-aux, and M-full vs each Step 15 ablation.

**Sensitivity analyses** (reported but not used for decisions):

1. Image-grouped bootstrap on pooled predictions: resample image groups 2 000 times within each repeat, paired across arms.
2. Exact sign test on the 15 fold differences.
3. Primary contrast restricted to repeat 1.
4. Primary contrast with TTA applied to both arms.
5. Primary contrast on EfficientNet-B0 (replication; the same decision rule is applied and reported separately).

**Power statement.** The corrected standard error is σ\_d · √(1/15 + 0.25) ≈ 0.56 σ\_d. For 80 % power at α = 0.05 (two-sided, df 14), the minimum detectable effect is ≈ 3.0 × 0.56 σ\_d ≈ 1.7 σ\_d. If σ\_d = 0.02, MDE ≈ 0.034; if σ\_d = 0.03, MDE ≈ 0.051. State the assumed σ\_d range (0.02–0.03) in the timestamped plan before Step 10, and report the observed σ\_d afterwards. Under the correction, extra repeats barely help: √(1/J + ρ) falls only from 0.56 to 0.54 with 25 folds. True effects between 0.02 and the MDE will therefore often read as "inconclusive", and the paper says so in advance.

**Output:** `table2_main.tex`, `table3_ablation.tex`, `fig_paired_differences.pdf` (one point per fold, coloured by repeat, with mean and corrected CI), `stats_log.json` (all test statistics, df and adjusted p-values).

## Steps 15–16. Ablations, quality gates and deviations

### Step 15. One-factor ablations

Each ablation changes one flag from the frozen M-full config. It is run on all 15 folds and compared with M-full under Family C.

| Ablation | Flag change | Question |
| --- | --- | --- |
| M-aux (Tier 1) | `use_boost=False` | Is the inference-time boost needed? |
| Boost without aux loss | `lambda_g=0` (gate trained only through CE on s′) | Is gate supervision needed? |
| Detached gate | `detach_gate=True` | Does the gain come from shaping the backbone? |
| + Coordinate Attention | `use_coordattn=True` | Does the attention block add anything? |
| Optional: + prototype bank, + mixup, + PCGrad (species + stage + gate), + minority injector | One flag each | Do the original tricks add anything? Run them only if the paper keeps any of them |

**Interpretation rule:**

- A component is said to "contribute" only if removing it lowers macro-F1 with a Holm-adjusted CI that excludes 0.
- Otherwise the wording is "no evidence that X contributes".
- Non-monotone ablation results are reported as they are.

### Step 16. Quality gates, stop rules and deviations log

**Quality gates.** Do not pass a gate until its check is met.

| Gate | After step | Must hold |
| --- | --- | --- |
| G1 Data frozen | 3–4 | Manifests hashed, QC complete, Table 1 and Table 5a written; label provenance and BBBC041 exposure statement recorded |
| G2 Splits frozen | 5 | `test_splits.py` passes; digests committed |
| G3 Code verified | 8 | All tests pass; tag `v1-code-verified` |
| G4 Configs and plan frozen | 9 | `v2-frozen` tag; grids and tie rules frozen; analysis plan timestamped (OSF or signed tag) |
| G5 Tier 1 complete | 10–11 | 45 runs complete and checked; only now is the primary analysis run |
| G6 All tiers complete | 10–12 | Completeness check passes for every tier run |
| G7 Numbers from code | 13–14 | Every number in the paper is traceable to `stats_log.json` or a generated table |

**Stop rules:**

- **Feasibility (Step 3):** a rare stage occurs in fewer than 5 image groups → use 3 folds; if still infeasible, stop (never merge stages under a four-stage endpoint), and decide before G2.
- **Nondeterminism (Step 1):** same-seed reruns differ by more than 0.02 macro-F1 → fix it, or report it and add two outer repeats (r4, r5) for Tier 1.
- **Leakage:** any failed disjointness or test-access check at any time → stop, fix it, and rerun every affected run from scratch.
- **Compute shortfall:** finish Tier 1 first. Report every incomplete tier as "not run"; never report partial folds.

**Deviations log (`logs/deviations.md`).** Add an entry for every change to this protocol after G4. Each entry gives:

- the date;
- the step;
- what changed and why;
- whether any test result had been seen at the time.

The paper's supplement reproduces the log in full.

## Step 17. Reporting, release and timeline

### Paper exhibits (all generated by `run_analysis.py`)

| Exhibit | Content |
| --- | --- |
| Table 1 | Data: images, image groups and cells per species × stage; exclusions |
| Table 2 | Fold coverage: rare-stage cells and groups per fold (summary; full table in the supplement) |
| Table 3 | Main result: B0–B6, B\*, M-aux and M-full. Columns: per-fold macro-F1 (mean ± SD), pooled schizont F1 and gametocyte F1, balanced accuracy, ECE, params and latency. Then the primary contrast row with d̄, corrected CI and the decision-rule verdict |
| Table 4 | Ablations relative to M-full, with Holm-adjusted CIs |
| Table 5 | BBBC041 external: data (5a) and results by arm and subset (5b) |
| Figure 1 | Model diagram of exactly what was run (gate, boost, loss terms; nothing that was removed) |
| Figure 2 | Paired per-fold differences, M-full − B\* and M-full − M-aux |
| Figure 3 | Pooled confusion matrices, B\* vs M-full (MP-IDB and BBBC041) |
| Figure 4 | Reliability diagrams after temperature scaling |
| Supplement | Tuning grids and scores; B\* choice; EfficientNet-B0 replication; sensitivity analyses; deviations log; the February 2026 runs labelled exploratory |

### Writing rules

- Give one name to the model that was run, and use it everywhere: in the text, figures and code.
- Keep regime labels on every metric (per-fold or pooled).
- Use the decision-rule wording verbatim.
- Report negative and inconclusive results in the abstract with the same prominence as positive ones.
- No claims about detection, NIH, "Gap 1/3", or deployment beyond the measured latency.
- Follow the CLAIM 2024 checklist and include it as a supplementary file.

### Release (at submission)

- [ ] Code at tag `v2-frozen` plus the analysis commit (GitHub, archived on Zenodo with a DOI)
- [ ] `cells.csv`, `bbbc041_cells.csv`, exclusions, QC and digests (no images; download scripts instead)
- [ ] All split JSONs and digests
- [ ] Frozen arm configs and tuning logs
- [ ] All prediction Parquet files and `stats_log.json`
- [ ] Best weights of the Tier 1 models (15 × 3)
- [ ] Timestamped analysis plan and deviations log

### Timeline (one GPU at ≈ 20 h/day; adjust to your hardware)

1. **Weeks 1–2:** Steps 1–5.
   - Label provenance is the long pole. Budget time for annotating or excluding multi-stage images by hand.
   - Exit: gates G1–G2.
2. **Week 3:** Step 8 code changes, tests and the GPU pilot → G3. Re-estimate compute from the measured run time.
3. **Week 3 (end):** freeze grids and the analysis plan, then timestamp them → G4.
4. **Weeks 4–5:** Tiers 1–2, with nested selection inside each fold (≈ 55 GPU-h planned) → G5. Then run the primary analysis.
5. **Week 6:** Tiers 3–5 and the external evaluation → G6.
6. **Week 7:** Steps 13–14 full analysis → G7. Generate all exhibits.
7. **Weeks 8–9:** write the stage-only paper (new title and scope). Then update thesis Chapter 7 by replacing its placeholders with the generated tables.
