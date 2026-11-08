PY ?= .venv/bin/python
BRAND ?= vaelo

.PHONY: install mock mock-sample abo seed discover tune classify decide draft submit run ui staff search eval test

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev,mock,ml,browser,ui]"
	$(PY) -m playwright install chromium

test:
	$(PY) -m pytest -q

# The targets below are wired up in later phases (SPEC §9).
mock:
	$(PY) -m uvicorn --factory mocksites.app:create_default_app --host 127.0.0.1 --port 8000

mock-sample:  # small demo catalog for browsing before `make seed` exists
	$(PY) -m mocksites.sample

abo:  # ABO metadata + selected images (cached in data/abo/) → seed/brand + seed/other_images
	$(PY) -m seed.import_abo build

seed:  # renders synthetic brand/other assets if missing, then 120 listings + seed/labels.json
	$(PY) -m seed.assets
	$(PY) -m seed.generate

discover:  # needs `make mock` running
	$(PY) -m bh.discovery.worker

tune:  # dev-split only: prefilter gate, own-photo routes, lookalike_clip (prints values for config.yaml)
	$(PY) -m eval.derive_prefilter
	$(PY) -m eval.tune_own_photo

classify:  # fits the dev-split scorer on the latest discovery run, then classifies its listings
	$(PY) -m eval.fit_scorer
	$(PY) -m bh.classify

decide:  # EV, action downgrades, review queue + budget, factual notice previews
	$(PY) -m bh.decide

draft:  # needs `make mock`: fill portal steps 1-4 for in-budget awaiting_approval cases (declarations untouched)
	$(PY) -m bh.enforce.portal

submit:  # needs `make mock`: sign + verify + submit cases a lawyer approved ("Approve and submit"); polls every 10 s
	$(PY) -m bh.enforce.submit --poll 10

run:
	@echo "run: implemented in Phase 8 (orchestrator)"; exit 1

ui:  # shoppers: search by name or photo, watch the agent, see which listings look genuine
	$(PY) -m streamlit run ui/app.py

staff:  # the brand's lawyers: review queue, approvals, human-required, cases, trace, evaluation
	$(PY) -m streamlit run ui/staff.py --server.port 8502

search:  # one shopper search from the command line, e.g. make search Q="vaelo bottle"
	$(PY) -m bh.shopper --search-id cli$$(date +%s) --text "$(Q)"

eval:
	@echo "eval: implemented in Phase 10 (evaluation)"; exit 1
