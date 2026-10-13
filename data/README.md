# data/

| Folder | Content | Tracked in git |
|---|---|---|
| `raw/` | untouched downloads (MP-IDB images + YOLO-seg labels, BBBC041, NIH) | no (download scripts / URLs recorded in `docs/protocol.md` Step 2 and 4) |
| `manifests/` | `cells.csv`, `cells_final.csv`, `crops.csv`, `bbbc041_cells.csv`, `digests.json`, `FROZEN.json`, exposure statement | **yes** once frozen (G1): copy the 34 frozen files here |
| `crops/` | 256 px native-resolution crops + masks, reproducible byte-for-byte from the manifest | no |
| `tables/` | Table 1, Table 5a, fold coverage, derived ratios, BBBC041 report | **yes** once frozen |
| `qc/` | `qc.csv` (decisions) tracked; contact sheets not tracked | partly |

The frozen manifests from the 12 Oct 2026 freeze (`FROZEN.json`, 34 files; 1,435 MP-IDB parasites, 2,452
BBBC041 cells) are not in this repository yet: copy `data/manifests/` and `data/tables/` from the frozen
project and run `python -m malprep gate` to confirm every digest.
