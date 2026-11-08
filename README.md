# Bounty Hunter

Agentic browser-use system that polices (mock) marketplaces for trade mark / copyright infringement of a
**fictional** brand, classifies what it finds against UK-law rules, prioritises cases, and drafts takedown
notices that a human lawyer approves before submission. See [SPEC.md](SPEC.md).

Decision support for a lawyer — not legal advice. Everything targets our own local mock sites.

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
make install                 # venv + dependencies + Playwright Chromium
cp .env.example .env         # set MOCK_PORTAL_USER / MOCK_PORTAL_PASS (and ANTHROPIC_API_KEY)
make seed                    # brand assets (if missing) + 120 listings + seed/labels.json
make mock                    # mock sites on http://127.0.0.1:8000
.venv/bin/python -m bh.browser.demo   # (other terminal) search both marketplaces, parse listings
```

`make test` runs the test suite. Later phases add `make ui` and `make eval`.

### Running the pipeline

With `make mock` running in another terminal:

```bash
make bootstrap               # first time: capture listings, fit the dev-split scorer, then a full run
make run                     # later runs: discover → classify → decide → draft (portal steps 1-4) → report
make submit                  # separate loop: signs + submits only the cases a lawyer approved
```

`make run` is `python -m bh.orchestrator.run --brand vaelo`, a LangGraph pipeline that writes everything to
one `Run` (tasks, tokens per model, wall time). Useful flags: `--llm` (borderline vision check and generic
extractor fallback), `--until discovery|classify|decide`, `--sites mockbay`, `--max-queries N`. If a portal
challenge pauses drafting, a human clears it in their own browser and then runs
`python -m bh.orchestrator.run --resume <run_id>`, which continues the same run without searching again.

On Windows, set `PYTHONUTF8=1` so the CLIs can print non-ASCII output.

### Real listings as background noise

To check the system on data it wasn't built around, mix real product listings into the mock sites:

```bash
make real-listings                                    # ~215 real ABO drinkware listings, all should be ignored
make run ARGS="--until decide"                        # same config and scorer as before
make eval ARGS="--labels seed/labels_real.json"       # adds a "real" row and the false-alarm rate
```

Titles, descriptions and photos are real (ABO, CC BY 4.0). Prices and sellers are generated, because ABO has
none. `make seed` restores the synthetic-only catalog.

### Seed images

Brand and listing photos come from the Amazon Berkeley Objects dataset (see Attribution):

```bash
make abo     # download ABO metadata (~105 MB) + the ~50 selected images into data/abo/, build seed assets
make seed    # regenerate listings and labels from them
```

The hand-reviewed subset is recorded in [seed/abo_selection.yaml](seed/abo_selection.yaml) (gold products show
no visible real branding; the fictional VAELO mark is composited on). Without the ABO assets, `seed/assets.py`
renders synthetic stand-ins so the project still runs offline. `python -m seed.check` reports hash and CLIP
separation of the generated data.

## Attribution

Product photographs are from the **Amazon Berkeley Objects (ABO)** dataset,
<https://amazon-berkeley-objects.s3.amazonaws.com/index.html>, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Collins, J., Goel, S., Deng, K., Luthra, A., Xu, L., Gundogdu, E., Zhang, X., Yago Vicente, T., Dideriksen, T.,
> Arora, H., Guillaumin, M., Malik, J. "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding."
> CVPR 2022.

Changes made: images were resized and padded or cropped to squares, and the fictional "VAELO" mark was
composited onto the gold product photos and onto counterfeit listing photos. The products shown are not Vaelo
products; Amazon does not endorse this project. Per-file credits: `seed/brand/ATTRIBUTION.md` and
`seed/other_images/ATTRIBUTION.md`.

"Vaelo", its marks and its UK registrations are fictional.
