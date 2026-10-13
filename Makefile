# HLG-Net confirmatory study: common commands. Run from the repository root.
PY ?= python
CONFIG ?= configs/data.yaml
UNIT ?= slide

.PHONY: install install-cpu test test-fast freeze-env arms plan check-runs analysis gate preflight smoke pilot queue status

install:              ## editable install of the four packages (+ test tools)
	$(PY) -m pip install -r requirements/dev.txt
	$(PY) -m pip install -e packages/malcore -e packages/malprep -e packages/malexp -e packages/malanalysis

install-cpu:          ## same, with the CPU-only PyTorch wheel (laptops / CI)
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(MAKE) install

test:                 ## all Step-8 and pipeline tests (CPU, ~3 min)
	$(PY) -m pytest

test-fast:            ## everything except the nested smoke run
	$(PY) -m pytest -k "not smoke_full_fold"

freeze-env:           ## Step 1: pin the environment
	$(PY) -m pip freeze > env/freeze.txt

arms:                 ## Step 9 / gate G4: write configs/arms/*.yaml + digests.txt
	$(PY) -m malexp arms-export --arms-dir configs/arms

plan:                 ## Step 9 / gate G4: timestamped analysis plan -> analysis/analysis_plan.md
	$(PY) -m malanalysis plan --config configs/analysis.yaml

gate:                 ## gates G1-G2 on the frozen data
	$(PY) -m malprep gate --config $(CONFIG)

preflight:            ## Step 8: one bf16 step of every arm variant on the GPU
	$(PY) -m malexp preflight --data-config $(CONFIG)

smoke:                ## Step 8: 2-epoch nested run of all arms on fold (r1, k0), tiny backbone
	$(PY) -m malexp smoke --data-config $(CONFIG) --unit $(UNIT)

pilot:                ## Step 8: GPU pilot (one full M-full run on the inner split of r1/k0)
	$(PY) -m malexp pilot --data-config $(CONFIG) --unit $(UNIT)

queue:                ## Step 10: runs/$(UNIT)/queue.csv
	$(PY) -m malexp queue --data-config $(CONFIG) --unit $(UNIT)

status:
	$(PY) -m malexp status --data-config $(CONFIG) --unit $(UNIT)

check-runs:           ## Step 11 completeness check (gate G5/G6)
	$(PY) -m malanalysis check --config configs/analysis.yaml

analysis:             ## Steps 13-14, 17: every number and exhibit -> analysis/
	$(PY) -m malanalysis run --config configs/analysis.yaml
