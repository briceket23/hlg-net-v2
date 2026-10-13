# Analysis plan (frozen before any confirmatory run; protocol Steps 13-14)

Timestamp: 2026-10-13T12:40:27  ·  commit: HEAD (dirty)
configs/analysis.yaml SHA-256: 3e8d54112f957b99735c4aa2de126435939ddd249aaf7599ed9410df6ffb1cea
malanalysis package SHA-256: 92b511f9ea2d5d1ef583d64ef238422e7fa12663ba91498ae799c979942dd6e6

## Primary endpoint and contrast
- Endpoint: per-fold four-class stage macro-F1 on the outer test fold (`zero_division=0`), 3 repeats x 5 folds.
- Contrast H1: M-full - B* per (r, k), B* = the best selected rebalanced baseline of each fold (inner validation).
- Test: corrected resampled t-test, rho = 0.25, df = J - 1, alpha = 0.05 two-sided.
- Decision rule (smallest effect of interest Delta_min = 0.02):
    - supported: "M-full improves stage macro-F1 over B* (H1 supported)"
    - below_delta: "Detectable but below the smallest effect of interest"
    - no_improvement: "No meaningful improvement"
    - inconclusive: "Inconclusive at this sample size"
    - worse: "M-full is worse than B*"

## Secondary contrasts (Holm within each family)
- Family A (H2), M-full vs B*: schizont F1 and gametocyte F1 pooled per repeat with grouped-bootstrap CIs on the repeat-averaged difference; balanced accuracy (corrected t); external macro-F1 (descriptive).
- Family B, the ladder: M-full vs each of B0, B1, B2, B3, B4, B5, B6.
- Family C (H3) and ablations: M-full vs M-aux, and M-full vs each of ABL-noaux, ABL-detach, ABL-coordattn.

## Sensitivity analyses (reported, not used for decisions)
1. Image-grouped bootstrap on pooled predictions (2000 resamples within each repeat, paired across arms, seed 2026).
2. Exact sign test on the paired fold differences.
3. Primary contrast restricted to repeat 1.
4. Primary contrast with TTA applied to both arms (only if TTA predictions are produced).
5. Primary contrast on EfficientNet-B0 (replication; same decision rule, reported separately).
6. Image-grouped splits (sensitivity for the slide grouping unit).

## Power statement
The corrected standard error is sigma_d * sqrt(1/15 + 0.25) = 0.56 sigma_d. For 80 % power at alpha = 0.05 (two-sided, df 14) the minimum detectable effect is 3.01 x 0.56 sigma_d = 1.70 sigma_d: sigma_d = 0.02 -> MDE = 0.034; sigma_d = 0.03 -> MDE = 0.051.
Assumed sigma_d range: [0.02, 0.03]. The observed sigma_d is reported afterwards. Under the correction, extra repeats barely help (sqrt(1/J + rho) falls from 0.56 to 0.54 with 25 folds); true effects between 0.02 and the MDE will often read as inconclusive, and the paper says so in advance.

## Rules
- Test predictions are opened only once a tier is complete (gate G5/G6); `malanalysis check` must pass first.
- Pooled values are labelled 'pooled' and never share a column with per-fold values.
- External macro-F1 always covers four stages; a stage with fewer than 30 cells is flagged low-support and its per-stage F1 shown with its interval.
- Every number in the paper is traceable to stats_log.json or a generated table (gate G7).
