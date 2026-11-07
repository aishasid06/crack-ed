PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)
BRAND ?= vaelo

.PHONY: install mock mock-sample abo seed discover tune classify decide draft submit run bootstrap ui eval test

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev,mock,ml,browser,orchestrator,ui]"
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

run:  # needs `make mock`: full LangGraph pipeline as one traced run (discover → classify → decide → draft → report)
	$(PY) -m bh.orchestrator.run --brand $(BRAND) $(ARGS)

bootstrap:  # first run on a fresh DB: capture listings, fit the dev-split scorer, then run everything
	$(PY) -m bh.orchestrator.run --brand $(BRAND) --until discovery
	$(PY) -m eval.fit_scorer
	$(PY) -m bh.orchestrator.run --brand $(BRAND)

ui:
	@echo "ui: implemented in Phase 9 (Streamlit UI)"; exit 1

eval:
	@echo "eval: implemented in Phase 10 (evaluation)"; exit 1
