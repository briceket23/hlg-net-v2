# runs/

`runs/<unit>/<arm>/<backbone-tag>/r{r}_f{k}/c{i}/` holds one candidate run:
`best.pt`, `config.yaml`, `fit.json`, `log.csv`, and, for the selected candidate only, `meta.json`,
`preds_test.parquet`, `preds_innerval.parquet`, `preds_bbbc041.parquet`.

`runs/<unit>/queue.csv` is the run matrix (Step 10) and `runs/<unit>/selection.csv` the nested
selection log (Step 9). Both are tracked; checkpoints and predictions are released separately
(Zenodo, Step 17).
