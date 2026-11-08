# Bounty Hunter — Implementation Spec

You are implementing the core of **Bounty Hunter**, a hackathon project: an agentic browser-use system that polices e-commerce sites for trade mark / copyright infringement of a brand, classifies what it finds, prioritises what is worth acting on, and drafts takedown submissions that a human lawyer approves before they are submitted.

Read this whole file before writing code. Work **phase by phase** (section 9). After each phase: run it, run the tests, fix failures, and summarise what works before moving on. Do not stub or fake functionality silently — if something is not implemented, raise `NotImplementedError` with a clear message. If you need to deviate from this spec, say why in your summary.

---

## 1. Hard constraints

- **Fictional brand only.** All brand assets, listings and sellers are synthetic. Never target real brands or real sellers.
  - *Narrow exception, real-brand demo (added 2026-11-08 at the project owner's request).* A brand folder may set `"fictional": false` only together with `"demo_real_brand": true`. It runs only with `--allow-real-brand-demo`, which:
    - forces `--until classify`;
    - allows read_only sites only;
    - requires explicit `--query` text.

    Every output carries the banner "Demo using publicly available product data. We don't represent this brand; no claim is made against any seller." (`bh.config.DEMO_BANNER`; also in `Run.config_snapshot.demo_banner` for the UI). `bh/enforce/guard.py` `check_case_claimable` refuses to decide, draft, approve or submit any case from such a run or about a non-fictional brand. Demo data must be public; trade mark registrations are never invented. Demo brands may be text-only (no photos). Purge with `python -m bh.discovery.purge --brand-dir <folder> --yes`. Tests: `tests/test_real_brand_demo.py`.
- **Submissions only go to our own mock sites.** Real marketplaces may only be read (optional, later). Nothing is ever submitted to a real site. Enforced in code: `bh/config.py` `ENFORCEMENT_HOSTS` (127.0.0.1, localhost) is the only set of hosts that portal login, drafting, review verification and submission may touch. Config `enforce.allowed_hosts` may narrow it, never widen it. Sites registered `mode: read_only` are refused by `bh/enforce` outright. Refusals are logged, raise `EnforcementBlocked`, and happen before any browser navigation (`bh/enforce/guard.py`).
- **No anti-bot evasion.** No CAPTCHA solving, fingerprint spoofing or proxy rotation. When a challenge page is detected, the task pauses with status `waiting_human`, the UI shows it, and the task resumes after a human completes it.
- **Human approval before every submission.** Before approval, enforcement fills portal steps 1–4 only (no declarations). The declarations are signed, and Submit clicked, only after the case is marked `approved` in the DB, with the approving lawyer (`Case.decided_by`) as signatory (see 7.1).
- **Secrets never enter LLM prompts.** Credentials come from `.env` and are typed by code, not by the model. Never log passwords or cookies.
- **Ground-truth labels are never visible to the agent.** The mock site must not render labels anywhere (no HTML comments, data attributes, alt text).
- **Log tokens, model and time for every LLM call and every task.** Judging is on workflow complexity, tokens used, model used and time to complete.
- **Legal rules are data, not model output.** Statutes, sections and required evidence elements live in `rules.yaml` (section 7.3). The LLM may assess whether an element is met and write reasoning, but must never invent a legal basis. Every case shows its rule id, citation and per-element evidence.
- **Notices stay factual.** Drafted notices state facts (URL, right relied on, registration number, evidence) and never threaten proceedings — unjustified threats of trade mark infringement can create liability (Trade Marks Act 1994, ss.21–21F). This is also why human approval is mandatory.
- **Not legal advice.** The UI labels outputs as decision support for a lawyer.

## 2. Tech stack

- Python 3.11+, `asyncio` throughout
- Browser: **Playwright (async)**, Chromium, persistent context per site (`.sessions/<site>/`)
- Orchestration: **LangGraph** (fan-out with `Send`), running as a **separate worker process** — never inside Streamlit
- Mock sites: **FastAPI** + Jinja2 templates, served locally
- DB: **SQLite** via **SQLModel**
- Images: `Pillow`, `imagehash` (perceptual hash), `sentence-transformers` with `clip-ViT-B-32` for embeddings
- Scorer: `scikit-learn` (`LogisticRegression` + `CalibratedClassifierCV`)
- LLM: Anthropic Python SDK; model names configured in `config.yaml` (a small model for navigation/extraction, a strong model for ambiguous classification and drafting)
- UI: **Streamlit** (reads/writes DB only)
- Tests: `pytest`, `pytest-asyncio`

## 3. Repository layout

```
bounty-hunter/
  SPEC.md
  config.yaml              # models, site limits, budgets, thresholds
  rules.yaml               # classification rules
  .env.example             # MOCK_PORTAL_USER, MOCK_PORTAL_PASS, ANTHROPIC_API_KEY
  bh/
    config.py
    db.py                  # SQLModel models + session helpers
    trace.py               # timing/token logging context managers
    llm.py                 # Anthropic wrapper that records tokens to trace
    browser/
      session.py           # persistent contexts, login(), challenge detection
      adapters/            # one adapter per site: search(), parse_results(), parse_listing()
        base.py
        mockbay.py
        tradeloft.py
      generic_extract.py   # LLM-based fallback extractor for unknown layouts
    discovery/
      queries.py           # query generation (templates + optional LLM)
      worker.py            # search → paginate → cards → dedup → prefilter → open listings
    features.py            # hash, CLIP, price, text, seller features vs gold assets
    classify.py            # rules.yaml + calibrated scorer + LLM fallback for borderline
    decide.py              # action, legal basis, expected-value priority
    enforce/
      portal.py            # login, multi-page form fill, upload evidence, stop at review
      submit.py            # submit only approved cases
    orchestrator/
      graph.py             # LangGraph pipeline
      run.py               # CLI entry: python -m bh.orchestrator.run --brand vaelo
  mocksites/
    app.py                 # FastAPI app mounting both marketplaces + report portal
    templates/
    static/
  seed/
    brand/                 # gold assets: photos, products.json, authorised_sellers.json
    generate.py            # builds listings + images + labels.json, seeds mock sites
    labels.json            # ground truth (never served)
  eval/
    evaluate.py            # join predictions to labels, metrics, PR curve, bootstrap CIs
    fit_scorer.py          # fit calibrated scorer on dev split
  ui/
    app.py                 # Streamlit
  tests/
```

## 4. Data model (SQLModel)

- **Brand**: id, name, rrp per product (via Product)
- **TradeMark** (mock registration): id, brand_id, mark_text, mark_type (word | figurative), registration_number, classes (JSON, e.g. [21] for drinkware), territory ("UK"), status ("registered"), logo_path
- **Product** (gold asset): id, brand_id, name, rrp, image_paths (JSON), phashes (JSON), clip_embeddings (path to .npy)
- **AuthorisedSeller**: id, brand_id, platform, seller_name
- **Listing**: id, platform, url (unique), external_id, title, description, price, currency, seller_name, seller_location, seller_age_days, rating, quantity, image_urls (JSON), local_image_paths (JSON), screenshot_path, discovered_by_query, first_seen_at
- **Features**: listing_id, matched_product_id, hash_dist, clip_sim, price_ratio, authorised, brand_in_text, misspelt_brand, replica_kw, inspired_kw, seller_age_days
- **Case**: id, listing_id, category, confidence, legal_basis (JSON list), action, expected_value, status, reasoning, created_at, decided_at, decided_by
  - category ∈ `authorised | genuine_resale | counterfeit_identical | counterfeit_similar | copied_photos | grey_import | modified_goods | lookalike | unrelated`
  - action ∈ `ignore | monitor | platform_report | escalate | refer_trading_standards`
  - also: rule_id, legal_elements (JSON: each required element → met/unmet/unknown + evidence string)
  - also: declared_by, declared_at — signatory (= decided_by) and time the portal declarations were completed
  - also: review_diff (JSON) — mismatches between the portal review page and the approved case, if any
  - status ∈ `classified | awaiting_approval | approved | rejected | drafted | submitted | waiting_human | failed`
- **Task**: id, run_id, type, site, dedup_key (unique per run), status, parent_id, attempts, started_at, finished_at, error
- **LLMCall**: id, task_id, model, purpose, input_tokens, output_tokens, latency_ms, created_at
- **Run**: id, brand_id, started_at, finished_at, config_snapshot (JSON), totals (JSON)

## 5. Mock sites (`mocksites/`)

Serve on `localhost:8000` with path prefixes. Deliberately make the two marketplaces structurally different so adapters genuinely differ.

**MockBay** (`/mockbay`)
- Search: `/mockbay/search?q=&sort=price_asc|relevance&page=N`, 12 cards/page, grid layout
- Listing page: `/mockbay/item/{id}` with image gallery, seller box, description
- A cookie-consent banner that must be dismissed

**TradeLoft** (`/tradeloft`)
- Search via a form with category dropdown and price filter, infinite-scroll style "Load more" button
- Listing page uses a different DOM structure (table of specs, seller in sidebar)
- Search-by-photo: `/tradeloft/visual-search` file upload; returns listings ranked by CLIP similarity server-side

**Rights-holder report portal** (`/portal`)
- Login page (credentials from `.env`), session cookie
- Multi-page IP report form:
  1. Rights owner details
  2. Right type (trade mark / copyright / both) + registration number, classes and territory for trade marks, or work description and first publication date for copyright (dynamic fields depending on choice)
  3. Infringing listing URLs (add-more button) and, per URL, the specific infringement (statute + section from the case) — required so the notice gives the host actual knowledge (E-Commerce Regulations 2002, reg.19)
  4. Evidence upload (screenshots)
  5. Declaration checkboxes + review page with **Submit**
- Store submissions in its own table; `/portal/submissions` lists them
- **Challenge page**: randomly (configurable probability, and after > N rapid requests) interpose `/portal/verify` with a simple "I am human" step that the agent must NOT solve — it detects it and hands off.

Ground-truth labels live only in `seed/labels.json` and are never read by `mocksites/` templates.

## 6. Seed data (`seed/generate.py`)

- Input: `seed/brand/` with 3–5 products, 3–4 photos each, `products.json` (name, rrp, optional currency (default GBP), goods class), `authorised_sellers.json`, and `trademark.json` (mock UK registration: mark text, word/figurative, registration number e.g. `UK00004123456`, classes, logo path). The registration is fictional and clearly labelled as such in the UI.
- Generate 120 listings across both marketplaces with this mix: authorised 15, genuine_resale 15, counterfeit_identical 15, counterfeit_similar 15, copied_photos 15, grey_import 10, modified_goods 10, lookalike 15, unrelated 10.
- Grey imports: genuine photos/near-RRP prices but seller outside UK/EEA or "imported from USA/Japan" wording. Modified goods: genuine photos plus "customised", "relabelled", "refurbished" wording.
- One seller (or a small cluster sharing images/text) gets ≥ `bulk_listing_threshold` counterfeit listings, to exercise the Trading Standards overlay.
- Image variants for copied/counterfeit: resize, crop, mirror, brightness, JPEG quality 25, watermark overlay.
- Lookalike/unrelated images: from `seed/other_images/` (photos the team takes).
- Titles/descriptions: template-based per category (with optional LLM generation behind a flag). Counterfeits include misspellings, "replica", "1:1", bulk quantity; lookalikes include "inspired by", "dupe"; resale includes "pre-owned", "used".
- Prices relative to RRP per category (counterfeit 0.1–0.45×, resale 0.6–0.95×, authorised 0.85–1.0×, etc.). Seller location/age/rating sampled per category.
- Include ~20% "hard" cases (tagged `difficulty: hard`).
- Write `labels.json` with `{id: {category, legal_basis, expected_action, should_report, difficulty, split}}`, 60/40 dev/test split stratified by category and difficulty.
  - `expected_action` ∈ `platform_report | escalate | monitor | ignore` is the action of the category's rule in `rules.yaml`; `legal_basis` is that rule's citation (plus TMA s.92 for the bulk counterfeit cluster).
  - `should_report = (expected_action == "platform_report")`.

## 7. Pipeline behaviour

### 7.1 Orchestrator (`orchestrator/graph.py`)
LangGraph nodes:
1. `load_brand` → loads gold assets, computes/caches phashes and CLIP embeddings
2. `generate_queries` → templates (brand, product names, misspellings, generic descriptors, "replica"/"inspired by" variants); optional single LLM call to extend
3. `fan_out_discovery` → `Send` one discovery task per (site, query); concurrency limited per site by `config.yaml` (asyncio semaphores)
4. `dedup_and_prefilter` → dedup by URL; download thumbnails; keep cards where `clip_sim ≥ prefilter_clip` OR `hash_dist ≤ prefilter_hash` OR brand fuzzy-matches title. Scoring happens page by page during search: a (site, query) task keeps paging until `discovery.stop_after_empty_pages` (2) consecutive pages contain no prefilter-passing card, with the site's `max_pages` as a hard safety cap
5. `fan_out_capture` → open each surviving listing in parallel; extract full data; screenshot; download images
6. `classify` → features + rules + scorer; LLM vision check only when confidence in borderline band
7. `decide` → action, legal basis, expected value; cases with action `platform_report` go to `awaiting_approval`
8. `draft` → for each awaiting case, log into portal, fill form steps 1–4 (owner, right, listing URL + statute/section, evidence) and stop. Declarations are **not** touched before approval. Status `drafted` (or `waiting_human` if a challenge appears)
9. `report` → write Run totals (tasks, sites, tokens per model, wall time)

A separate loop `enforce/submit.py` polls for `approved` cases. Approval ("Approve and submit") is the single human authorisation. For each approved case the loop:

1. resumes the saved draft and completes the declarations (step 5) with `Case.decided_by` as the signatory, recording `declared_by` / `declared_at`;
2. on the review page, verifies the page against the approved case: listing URL(s), right type, registration number, statute/section per URL, and evidence count;
3. if everything matches, clicks Submit and stores the portal reference (status `submitted`). On any mismatch it does **not** submit: the case goes to `waiting_human` and the field-by-field diff is stored in `Case.review_diff` for the UI.

If `enforce.require_final_confirm` is true (default false), the loop stops after a successful verification and waits for a second confirmation in the UI before clicking Submit. Rejected cases are marked and never get declarations or a submission.

The worker writes everything to the DB; Streamlit only reads and writes approval decisions.

### 7.2 Browser layer
- Site registry (`config.yaml` `sites`): `base_url`, concurrency, pacing, hard `max_pages` cap, `adapter` (null = no adapter), `search_url` (optional), and `mode: full | read_only`. A `read_only` site may be searched and extracted but never logged into or submitted to.
- `session.py`: one persistent context per site; `login(site)` fills credentials from env (never via LLM); `detect_challenge(page)` checks for the challenge page and raises `HumanRequired`.
- Adapters implement deterministic `search(query, page_no)`, `parse_results(page) -> list[Card]`, `parse_listing(page) -> ListingData` using selectors.
- `generic_extract.py`: fallback that sends a distilled accessibility-tree snapshot (not raw HTML, not screenshots by default) to the small model and asks for JSON matching a Pydantic schema. Used when an adapter's selectors fail; record that the fallback was used.
- Retries: 1 retry with fresh page, then fallback extractor, then mark task `failed` with reason.

### 7.3 Classification — grounded in UK law (`rules.yaml` + `classify.py`)

Rules are evaluated in order; the first rule whose `requires` elements are all met wins. Each rule carries its legal citation and the evidence elements it needs. The calibrated scorer then produces `confidence` for `should_report`.

**Legal background the rules encode** (store the summaries in `rules.yaml`; do not hard-code them in prompts):

| Area | Law | Effect on rules |
|---|---|---|
| Trade mark infringement | Trade Marks Act 1994 (TMA) s.10 | s.10(1): identical sign + identical goods, confusion need not be shown. s.10(2): identical/similar sign + identical/similar goods + likelihood of confusion. s.10(3): reputed mark, unfair advantage or detriment. |
| Lawful resale | TMA s.12; UK+ exhaustion regime (made permanent 2025) | Rights in genuine goods placed on the UK or EEA market with consent are exhausted in the UK → resale is lawful. Exception (s.12(2)): goods changed or impaired. |
| Grey imports | TMA s.10 with s.12 not applying | Genuine goods first marketed outside the UK/EEA can still infringe in the UK. |
| Copied photos | Copyright, Designs and Patents Act 1988 (CDPA) | Product photos are artistic works; copying / communicating them to the public infringes. |
| Look-alikes | Passing off (common law); UK registered/unregistered design rights | Requires goodwill + misrepresentation + damage, or a protected design → human judgement. |
| Platform notices | Electronic Commerce (EC Directive) Regulations 2002, reg.19 | Hosts lose protection once they have actual knowledge and fail to act → notices must identify the exact URL, the right relied on and the owner. |
| Threats | TMA ss.21–21F | Unjustified threats create liability → factual notices, human approval. |
| Criminal counterfeiting | TMA s.92 | Large-scale counterfeiting → `refer_trading_standards`. |

**Rule set** (in order):

| rule_id | Category | Legal basis | Required elements | Action |
|---|---|---|---|---|
| `authorised` | authorised | — | seller in authorised list | ignore |
| `unrelated` | unrelated | — | no image match AND no brand/mark in text | ignore |
| `copied_photos` | copied_photos | CDPA 1988 | own photo (hash match) AND brand/mark not used for the goods | platform_report |
| `tm_identical` | counterfeit_identical | TMA s.10(1) | exact mark in text or logo match; goods in registered class; not authorised; not exhausted* | platform_report |
| `tm_similar` | counterfeit_similar | TMA s.10(2) | similar sign (misspelling / altered logo); same or similar goods; confusion indicator (own photos, get-up, "official" wording); not authorised; not exhausted* | platform_report if confidence ≥ act_threshold, else escalate |
| `modified_goods` | modified_goods | TMA s.12(2) | genuine indicators AND modification keywords ("customised", "relabelled", "refurbished", "repackaged") | escalate |
| `grey_import` | grey_import | TMA s.10 (exhaustion not applying) | genuine indicators AND origin outside UK/EEA (seller location or "imported from" wording) | escalate |
| `exhausted_resale` | genuine_resale | TMA s.12 (UK+ exhaustion) | own photo or brand; price_ratio ≥ resale_min_ratio; no replica keywords; no modification keywords; origin UK/EEA or unknown | monitor |
| `lookalike` | lookalike | Passing off / design rights | clip_sim ≥ lookalike_clip, no hash match, and (inspired_kw OR no brand) | escalate |
| fallback | unrelated | — | — | ignore |

\*"not exhausted" = any of: price_ratio < counterfeit_price_ratio, replica keywords ("replica", "1:1", "AAA", "mirror quality"), misspelt mark, seller cluster flagged as counterfeit.

Overlay: if a `counterfeit_*` seller has ≥ `bulk_listing_threshold` listings or bulk quantities, add a second recommended action `refer_trading_standards` (TMA s.92).

**`rules.yaml` format:**

```yaml
- id: tm_identical
  category: counterfeit_identical
  legal_basis:
    statute: "Trade Marks Act 1994"
    section: "s.10(1)"
    summary: "Identical sign used for identical goods; likelihood of confusion need not be shown"
  requires:
    - mark_exact_in_text_or_logo_match
    - goods_in_registered_class
    - not_authorised_seller
    - not_exhausted
  action: platform_report
  min_confidence: 0.85

- id: exhausted_resale
  category: genuine_resale
  legal_basis:
    statute: "Trade Marks Act 1994"
    section: "s.12 (UK+ exhaustion)"
    summary: "Rights exhausted once genuine goods are placed on the UK or EEA market with consent"
  requires: [own_photo_or_brand, price_ratio_ge_resale_min, no_replica_kw, no_modification_kw, origin_uk_eea_or_unknown]
  action: monitor
```

Each element name maps to a Python predicate in `classify.py` returning `met | unmet | unknown` plus an evidence string (e.g. `"price £6.99 = 25% of RRP £28.00"`). Store the result in `Case.legal_elements` and render it in the review queue, e.g.:

> **s.10(1) TMA** — identical sign ✓ · registered class 21 ✓ · not authorised ✓ · not exhausted ✓ (25% of RRP, "1:1 replica")

**Borderline handling:** if confidence falls in the borderline band (e.g. 0.4–0.7) or any required element is `unknown`, make one strong-model vision call with: gold image, listing image, extracted fields, and the candidate rules (with their `requires` lists). The model returns JSON `{rule_id, elements: {name: {status, evidence}}, reasoning}` and may only choose a `rule_id` from the provided list.

Thresholds (`counterfeit_price_ratio`, `resale_min_ratio`, `lookalike_clip`, `act_threshold`, `bulk_listing_threshold`, borderline band) live in `config.yaml`.

### 7.4 Decision (`decide.py`)
- `harm = price × max(quantity_or_reviews_proxy, 1) × clip_sim`
- `p_success` per (platform, legal_basis) from `config.yaml` priors
- `cost` = fixed lawyer-minutes per action, converted to the same units
- `expected_value = harm × p_success × confidence − cost`
- Action comes from the matched rule in `rules.yaml` (section 7.3), downgraded to `escalate` if confidence < the rule's `min_confidence` or any required element is `unknown`. The Trading Standards overlay can add `refer_trading_standards` alongside the primary action.
- Review queue ordered by expected value. Optional: given `review_budget_minutes`, select cases greedily by EV / cost.
- Approval records `decided_by` / `decided_at`. "Approve and submit" is the single human authorisation and the consent to sign the portal declarations in `decided_by`'s name, so the approval screen shows the exact declaration text before the button.

### 7.5 Tracing
`trace.py` provides `async with task_span(run_id, type, site, dedup_key) as task:` and `llm.py` records every call to `LLMCall`. Run totals aggregate tokens by model, task counts by type/status, max parallelism, wall time.

## 8. Streamlit UI (`ui/app.py`)

Pages:
1. **Run** — choose brand, start a run (spawns the worker process), live task counts by status, elapsed time, tokens by model (auto-refresh).
2. **Review queue** — cases `drafted`/`awaiting_approval` ordered by EV: gold image vs listing image side by side, extracted fields, category, confidence, the legal citation with a per-element checklist (✓ / ✗ / ? with evidence), reasoning, screenshot, and a preview of the factual notice text, and the full declaration text that will be signed in the approver's name; **Approve and submit** / **Reject** buttons write to DB. Cases blocked by a review-page mismatch show the stored diff. Banner: "Decision support for a lawyer — not legal advice."
3. **Human required** — tasks in `waiting_human` with a link to the page; "Done" button resumes the task.
4. **Cases** — filterable table of all cases and statuses.
5. **Trace** — task tree (parent/child), per-task timing, LLM calls with tokens/model.
6. **Evaluation** — output of `eval/evaluate.py`: discovery recall (separately from classification), confusion matrix, **precision on `should_report`** and **recall on `escalate`** reported separately, PR curve with chosen threshold, bootstrap 95% CIs, metrics by site and by difficulty.

## 9. Phases (implement in this order)

1. **Scaffold**: repo layout, config loading, DB models, trace + llm wrappers, `.env.example`, pytest setup. *Done when* `pytest` passes and DB creates tables.
2. **Mock sites**: MockBay, TradeLoft, portal (login, multi-page form, upload, review, submissions list, challenge page). *Done when* each page works manually and a test hits every route.
3. **Seed generator**: brand assets loader, listing generation, image variants, `labels.json`, seeding into mock sites. *Done when* 100 listings browsable, labels split 60/40.
4. **Browser layer + adapters**: sessions, login, challenge detection, both adapters, generic fallback extractor. *Done when* a script searches both sites and returns parsed cards and listings.
5. **Discovery pipeline**: queries, fan-out with per-site limits, dedup, prefilter, capture. *Done when* a run stores listings, screenshots and images in DB.
6. **Features + classification + scorer**: `features.py`, `rules.yaml` (data written in Phase 3; predicates here), `classify.py`, `eval/fit_scorer.py`. Own-photo matching uses multi-variant gold hashes (mirrored, multi-scale crops) and three routes: (1) pHash **and** dHash agree, (2) CLIP similarity ≥ a threshold **tuned on the dev split only** (maximise copied-photo recall subject to near-zero lookalike false matches; not fixed in advance), (3) ORB/SIFT keypoint matching for crops and resizes. wHash is never a route on its own. The own-photo element must not be decisive for counterfeit rules. *Done when* classification runs on all captured listings.
7. **Decision + drafting + submission**: `decide.py`; drafting fills portal steps 1–4 only; submit-on-approval loop completes declarations as `decided_by` (recording `declared_by`/`declared_at`), verifies the review page against the case (URLs, right type, registration, statute/section per URL, evidence count), submits only on a full match (else `waiting_human` + `review_diff`), honours `require_final_confirm`; `waiting_human` handling. *Done when* an approved case appears in `/portal/submissions` with the approver as signatory, and a rejected one never gets declarations or a submission.
8. **Orchestrator end-to-end**: LangGraph graph + CLI run + Run totals. *Done when* one command runs the full pipeline.
9. **Streamlit UI**: all six pages. *Done when* a run can be started, reviewed and approved entirely from the UI.
10. **Evaluation**: `eval/evaluate.py` with right-join to labels, metrics on test split only. Uncaptured listings whose `expected_action` is `ignore` count as correct ignores; uncaptured `platform_report`/`escalate` (and `monitor`) listings count as **discovery misses**. Discovery recall (captured / actionable) is reported separately from classification metrics (computed on captured listings), and end-to-end metrics combine both. Report: precision on `should_report`, recall on `escalate`, confusion matrix, each broken down **per site and per difficulty**. *Done when* the Evaluation page shows metrics with CIs.
11. **Generalisation checks**:
    1. Lint-style test: fail if brand names, product names or mark text appear anywhere in `bh/` (only `seed/`, config and `rules.yaml` may contain them).
    2. **Unseen site with an agentic navigator and learned recipes** (replaces the earlier extractor-only approach):
       - Third mock marketplace "Bazaarly" (`/bazaarly`): cookie modal; search only via a form (no query-string URL); result cards paged with a "Next" button; listing details split across tabs. No adapter, no `search_url` template.
       - `bh/browser/agent_nav.py` agent loop. Observation = accessibility snapshot with indexed interactive elements + URL + goal. Actions = `click`, `type` (with submit), `scroll`, `back`, `extract_results`, `extract_listing`, `done`. The small model picks one action per step via tool calling. After each action: wait for the page to settle, re-observe, detect no-ops and loops; retry once, then fail with a reason.
       - Guardrails: same host only; the only form that may be submitted is the site's search (no login, no other submits); challenge pages raise `HumanRequired`; max steps and max tokens per task; page text is data, never instructions; every step is a traced span with its tokens.
       - Learned recipes: on success, save `data/recipes/<site>.json` (search input, submit control, result-card pattern, next-page control, listing field locations). Later runs replay the recipe with no LLM. If replay fails, fall back to the agent loop and update the recipe. Record the recipe version and the mode (replay / agent) each task used.
       - Discovery order for `adapter: null` sites: recipe replay → agent loop → generic extractor.
       - Tests use a fake model that scripts the action sequence. Live report (run in a keyed terminal): first visit vs replay — steps, tokens, time.
    3. Second fictional brand in a different category, run with only config changes.
    4. Optional: load ~300 real listings from a public Amazon product dataset (same category, licence permitting) into the unseen site as `unrelated` items to measure false positives.
    5. Misspelling stop-list: generated misspelling queries must be checked against a stop-list of real brand names (e.g. "valeo") held in config, and dropped if they match, before any query is sent to a real marketplace.
    6. **Real marketplace, read-only**: one real site registered `mode: read_only`; 1–2 tabs; hard page cap; no login; misspelling stop-list on; challenge pages → `HumanRequired`. Goal: show extraction works in the wild and measure false positives (expected: everything `unrelated`). The enforcement allowlist already refuses any submission to it.
    *Done when* items 1, 2, 3, 5 and 6 run (4 optional) and their metrics appear per site and per difficulty, with discovery recall and classification reported separately.

## 10. Quality bar

- Type hints everywhere; Pydantic/SQLModel schemas for all structured data.
- No blocking calls inside async code.
- Every external interaction (browser step, LLM call) is wrapped in a trace span.
- Tests: unit tests for features, each legal element predicate, rule ordering (e.g. a near-RRP genuine listing from an unauthorised UK seller must classify as `genuine_resale`, never counterfeit), decision maths; integration test that runs the pipeline against the mock sites with LLM calls mocked.
- A `Makefile` or `justfile` with: `mock`, `seed`, `run`, `ui`, `eval`, `test`.
- README with setup and a 5-command quickstart.

## 11. Decisions log

- Portal challenge tracking is per client IP: a human passing `/portal/verify` in their own browser releases the paused agent on the same machine.
- The mock sites keep their own DB (`data/mocksites.db`); the catalog (`data/mocksites/catalog.json`) hot-reloads.
- `trademark.json` holds two fictional registrations (word mark + figurative logo mark).
- Labels carry extra fields (`site`, `url_path`, `product`, `bulk_cluster`, `image_sources`), never served to the agent.
- Seed images come from the Amazon Berkeley Objects dataset (CC BY 4.0, attribution in README and `seed/*/ATTRIBUTION.md`): a hand-reviewed subset (`seed/abo_selection.yaml`, built by `seed/import_abo.py`, cached in `data/abo/`). Gold products show no real branding and get the fictional mark composited on (`seed/mark.py`). Counterfeit listing photos are other same-category products with the mark printed on. The synthetic renderer (`seed/assets.py`) remains the offline fallback. Non-gold images live in `seed/other_images/{lookalike,unrelated,blank}/` with an `index.json` each.
- Measured on the ABO set (`python -m seed.check`): pHash alone confuses some lookalikes with gold photos (same backdrop/shape). dHash separates the white-background cases. wHash is unreliable on white studio shots (unrelated items as close as 4). CLIP: copied photos median 0.93, lookalikes max 0.854.
- Own-photo matching (Phase 6): first route pHash AND dHash agree; second route CLIP with the threshold tuned on the dev split only (maximise copied-photo recall with near-zero lookalike false matches), not fixed at 0.95; third route ORB/SIFT keypoint matching for crops/resizes; wHash never counts alone. Own-photo is never decisive for counterfeit rules.
- Approval is the single human authorisation for submission ("Approve and submit"). The agent verifies the review page against the approved case before clicking Submit; any mismatch blocks submission (`waiting_human` + stored diff). `enforce.require_final_confirm` (default false) adds a second confirmation.
- In-page browser actions (clicks, form submits, Load more, cookie dismiss, login submit) are trace spans, like navigations.
- Labels use `expected_action` / `should_report` instead of `should_act`.
- Prefilter CLIP gate derived from the **dev split only** (`python -m eval.derive_prefilter`): gate = floor₀.₀₁(min thumbnail CLIP similarity, over dev listings with `expected_action ≠ ignore` that the pHash and brand-text routes miss, − 0.02 margin) = **0.73** (4 dev listings needed the CLIP route; lowest a lookalike at 0.755). It replaced 0.65, which had been chosen from all data (leakage). On re-run, discovery recall stayed 45/45 report and 35/35 escalate (test split 18/18 and 14/14).
- Discovery paging is adaptive: stop after 2 consecutive pages with no prefilter-passing card; `max_pages` (MockBay 10, TradeLoft 15) is a hard cap only. On the 120-listing mock catalog the early stop never triggers (every page has a relevant card), so recall and cost are unchanged; it matters on larger, noisier result sets.
- Phase 6 thresholds, all derived from the **dev split only** and recorded here:
  - own-photo routes (`python -m eval.tune_own_photo`, image level; positives = dev images derived from gold photos, hard negatives = dev lookalike images, objective = max recall with zero lookalike matches): pHash ≤ 10 AND dHash ≤ 8 on the same gold variant; CLIP ≥ 0.854 (dev lookalike max 0.844 + 0.01); ORB ≥ 16 RANSAC inliers. Dev: 91/91 gold-derived images matched, 0/15 lookalikes.
  - `lookalike_clip` = 0.758: midpoint between the highest dev unrelated listing (0.726) and the lowest dev lookalike listing (0.79), listing level.
  - Scorer: `python -m eval.fit_scorer` (CalibratedClassifierCV(LogisticRegression, sigmoid, 3-fold), dev listings only).
  - These values are specific to the ABO image set. ORB counts in particular do not transfer (synthetic lookalikes reach ~41 inliers), so re-run the tuners whenever the seed images change.
- Seller clustering links sellers by normalised name or by shared *seller-specific* images. Unmodified copies of the brand's photos (pHash ≤ 2 to a gold photo) never link sellers, because resellers share them legitimately.
- Seed generator: grey importers have their own seller pool (they previously shared handles with counterfeit traders, an unrealistic artefact that made the seller-cluster signal mislabel them).
- Known limitation: a lookalike photographed in the same scene as a gold photo (same backdrop, similar product) matches all three own-photo routes and is classified `copied_photos`. It occurs only in the test split, so it was not tuned away.
- Known limitation: modification keywords match official product copy. In run 28 (real-brand demo on the Oxylabs sandbox), "personalized" in a publisher's description of in-game features classified Mario Kart Wii as `modified_goods`. The rule is unchanged (tuning happens on dev only). Such cases are surfaced instead by the "model disagrees" flag: when the borderline check's assessment contradicts the kept rules' outcome (a different rule, or an element of the kept rule marked unmet), the category stays and `Case.model_disagrees` / `model_disagreement` record the model's reason. `bh.show_cases`, `bh.decide.review_queue` and `python -m bh.enforce.approvals queue` show "model disagrees — review".
- Prices are compared and quoted in their own currency (run 28 found EUR listings treated as GBP). `products.json` takes an optional `currency` per product (ISO 4217; default GBP). `price_ratio` is computed only when the listing and the RRP are in the same currency (never converted), and is otherwise unknown with the evidence "not compared … (different currency)". Evidence, the borderline payload (`price` / `rrp` with `currency`) and notices use the listing's own currency symbol, and the borderline prompt forbids changing a currency.
- **Deferred: fails safe — same-scene own-photo trigger.** Found on the **test** split (3 listings using one hammered-copper photo shot in the same scene as gold `arc-copper_3`: ORB 149 inliers, pHash 6 / dHash 6, CLIP 0.854). Those cases are already *escalated*, not reported (scorer confidence below `copied_photos` `min_confidence` 0.80; test report precision 1.00), so the fix would only change the category label. It touches shared files and needs regenerate / re-tune / re-run, so it is deferred. Design for when it is picked up:
  1. Product boxes for gold photos as brand data: `seed/import_abo.py` writes `products.json` `image_boxes` (0–1 coordinates, from the placement override box or `object_box()`); `ProductAsset.image_boxes`, `GoldImage.box`. The synthetic fallback has no boxes, so the check is unavailable there.
  2. `OwnPhotoMatcher`: map the best ORB match's RANSAC inliers into gold coordinates and count those inside the product box (mirrored for mirrored variants) → `orb_product_inliers`, `orb_product_fraction`. `own_photo_conflict` = an own-photo route met, ORB evidence present, and `orb_product_fraction < own_photo.min_product_fraction` (match mostly on the background).
  3. Classification: on conflict, `own_photo` and `no_hash_match` become `unknown` with evidence ("N ORB inliers, only X% on the product: scene match"). `copied_photos` is then blocked, so the borderline vision check chooses among candidate rules. Without an LLM, the provisional outcome is the first blocked rule, and `decide.py` escalates because an element is unknown. Check that no non-conflict dev case changes; if one does, restrict this to own-photo conflicts.
  4. Dev-only validation data: dev has no same-scene case. Add 3 hard dev lookalike listings from unused ABO photos that share a gold scene: `61NDIERgPsL` (hammered copper, dark backdrop of `arc-copper_4`), `81IIIj+5FEL` (copper jug, same backdrop), `91Dr4w8ZzSL` (smaller glass bottle in `clear-glass_4`'s laptop scene). New `same_scene_dev` pool in `abo_selection.yaml` (visual check first), appended after the main 120 with a separate RNG and excluded from the split, so existing listings, ids and the dev/test assignment are unchanged.
  5. Tuning, dev only: choose `min_product_fraction` to separate dev gold-derived ORB matches from dev same-scene matches; re-tune route thresholds with the conflict applied (zero lookalike own-photo matches after the conflict check). Then re-run discovery, fit the scorer and the full pipeline, and evaluate on test.
- Notices state plain facts only: listing title, seller and price; the sign used (identical, or similar, to the registered mark) for goods in the registered class, or that the photographs are reproduced from the owner's photography; and replica wording if present. The classifier's analysis (price/RRP, seller clusters, image routes) stays in the case file for the lawyer. The same text fills the portal's per-URL infringement field, and the review-page check compares it too, so a stale or edited draft blocks submission.
- Notices are generated from a template, not by the LLM. That deviates from §2's "strong model for drafting", deliberately: a template guarantees factual, non-threatening wording (`assert_factual`).
- Rights-owner details (fictional) live in `seed/brand/brand.json` (`owner`, `copyright`), so `bh/` stays brand-neutral.
- Drafting is limited to cases selected within `review_budget_minutes` (greedy EV per lawyer-minute; `enforce.draft_only_in_budget`). The review queue ranks every awaiting_approval and drafted case by EV.
- A portal challenge during drafting pauses that case (`waiting_human`) and stops the drafting loop, because the per-IP challenge would also hit later cases. The interrupted portal draft is abandoned, and the case is redrafted from scratch after `resume_case`.
- Orchestrator (Phase 8): graph state holds only JSON-safe values (ids, queries, per-task summaries). Live resources (browser, encoder, LLM client, prefilter cache, per-site semaphores) are passed as LangGraph runtime context. Discovery and capture fan out with `Send`; the per-site semaphore is taken before a task's span opens. Prefilter scoring stays page by page inside each search (per §7.1 node 4), so the `dedup_and_prefilter` node applies the listing budget and picks what to capture.
- One command produces one `Run`: the stage CLIs (`bh.discovery.worker`, `bh.classify`, ...) still open their own runs, but the orchestrator threads a single run id through every node.
- Resume: `--resume <run_id>` reopens the same run. Succeeded tasks are skipped by dedup_key; discovery tasks store their kept cards and capture tasks their listing id, so nothing is searched again. Classification is skipped when every listing already has a case. Decide and draft are idempotent and re-run. Resuming releases this run's drafts paused by a challenge (`resume_case`). Tasks a crashed worker left `running` are requeued. When a task is retried, browser step spans beneath it that are still `waiting_human` are marked failed ("superseded").
- `--until discovery|classify|decide` cuts the graph short (report always runs). The first run on a fresh DB needs `--until discovery` before `eval.fit_scorer`, because classification requires a fitted scorer; preflight checks this, the brand, portal credentials and site reachability before any browsing.
- LLM calls are opt-in (`--llm`), as in the stage CLIs. Without it, borderline cases are noted "LLM check not configured" and no tokens are spent.
- Evaluation (`python -m eval.evaluate`, `make eval`): reads the cases of one run (default: latest run with cases), right-joins `seed/labels.json` by URL path, test split by default.
  - Report precision / recall: predicted `platform_report` vs `should_report`.
  - Escalate recall: headline is lenient (predicted escalate or platform_report, i.e. the listing reaches a lawyer); strict (exactly escalate) is reported alongside.
  - Discovery recall: captured / actionable. Uncaptured `ignore` listings count as correct ignores.
  - End-to-end action accuracy treats uncaptured listings as ignore.
  - Intervals: Wilson score 95% for proportions (the percentile bootstrap collapses at 0/n and n/n) and bootstrap percentile 95% (2000 resamples, seed 0) for accuracies; `metrics.json` carries both.
  - Outputs: `data/eval/run_<id>_<split>/metrics.json` (schema_version 1, read by the UI's Evaluation page) and `report.md`.
- Task status `superseded`: a step span (e.g. a login or navigation that hit a portal challenge) whose parent task was retried after the human hand-off. The retry replaces it, so it is not an error. Run output, the evaluation report and the UI must show it neutrally, e.g. "2 superseded by human hand-off" (`bh.trace.describe_task_statuses`), never as a failure. Earlier runs had these recorded as `failed`; they were migrated in place (backup `data/bounty_hunter.<timestamp>.bak.db`).
- Enforcement allowlist and `read_only` site mode were implemented early (before Phase 11) because they back a §1 hard constraint. Tests (`tests/test_guard.py`): config widening is rejected; a non-allowlisted portal host and a `read_only` portal are refused, with a log entry and no browser navigation, by drafting, submission, the CLIs and portal login; off-host page URLs block form submission.
- Brand folders: `python -m bh.orchestrator.run --brand-dir <path>` protects the brand in any folder with the `seed/brand/` layout (e.g. UI-created `data/brands/<slug>/`); `bh.brand.validate_brand_dir(path)` lists every error and warning without raising. Rules:
  - `fictional: true` is required (§1).
  - Photos and logo must be JPEG or PNG, ≤ 10 MB, and ≥ 300 px on the shorter side; image paths must stay inside the folder (no `..`, no absolute paths, no symlink escape).
  - The folder path must not contain `..` and must resolve to a directory.
  - `trademark.json` is optional (copyright-only brand: trade mark rules cannot apply, and trade mark notices are not reportable).
  - A registration number may belong to one brand only.
  - Each run records its brand and `brand_dir` in `Run.config_snapshot`; decide, drafting and the submit loop load the brand of each case's run (`config_for_run`, `BrandsByRun`), so one submit loop can serve several brands. `--resume` reuses the run's brand.
- The seed logo is rendered at 1050×400 so seed brands meet the 300 px rule.
- Real marketplace, read-only (Phase 11 items 5 and 6): the safeguards are implemented, but no real site is registered yet. Choosing one is a human decision: its terms of service and `robots.txt` must allow automated reading. `config.yaml` has a commented template. Enforced in code:
  - Any site on a real host (not `ENFORCEMENT_HOSTS`, not a reserved TLD such as `.test`) must be `mode: read_only`. `read_only` sites are capped at `max_concurrency` 1, `max_pages` ≤ 2 and `min_delay_ms` ≥ 2000 (`bh/config.py`). `BrowserManager` re-validates every site, so a loosened `model_copy` cannot open a browser.
  - Misspelling stop-list: `discovery.real_brand_stoplist` (shipped with `valeo` and `velo`) is required when a `read_only` site exists. A query containing a stop-listed name as a whole word is never sent to a `read_only` site: the discovery task records `stop: stop_list`, and `iter_search_pages` raises `ReadOnlyBlocked` as a backstop.
  - `robots.txt` is honoured on `read_only` sites (`bh/browser/readonly.py`) for `browser.robots_user_agent` or `*`. A 5xx, 401/403 or unreachable `robots.txt` means read nothing; 404 means no rules. The check runs before every `goto`, before a navigator or replay click on a link or a form submit (including the cookie-consent form), and on the page landed on (otherwise go back and stop). Preflight checks the base URL.
  - `extra_hosts` (read_only only) lists other hosts of the same site the navigator may read, e.g. `m.` or `www.` variants.
  - No case about a real seller: preflight refuses `--until decide|draft` when a `read_only` site is selected (false positives are measured with `--until classify`). `check_listing_target` (`bh/enforce/guard.py`) refuses drafting, submission (`load_case`) and approval of any listing from a `read_only` site or on a host outside the allowlist.
  - Data hygiene: `python -m bh.discovery.purge --site <name> [--yes]` deletes a `read_only` site's listings, features, cases and listing folders and redacts its trace payloads. Mock sites cannot be purged.
  - Tests (`tests/test_readonly.py`) run on the mock server with a site registered `read_only`; no real site is contacted.

- Real listings as background noise (`seed/real_listings.py`, `make real-listings`): 215 English-titled ABO drinkware listings (CC BY 4.0; no item used for seed photos) added to the seeded catalog, all labelled `unrelated` / `ignore`, `difficulty: real`, `source: abo`, split 60/40. Titles, bullet-point descriptions and photos are real; price (£4–45), seller, location, age and rating are drawn from one neutral range, because ABO has none. Labels go to `seed/labels_real.json`; `seed/labels.json` stays the synthetic-only baseline. ABO has only ~215 usable drinkware items (mostly glasses and cups, ~14 bottles/flasks), fewer than the ~300 planned.
- Evaluation adds `false_alarm_rate`: expected-ignore listings sent to a lawyer (reported or escalated) / expected-ignore listings.
- First real-noise result (run 4, same config and dev-fitted scorer as run 3, test split): synthetic metrics unchanged (report precision 16/16, escalate recall 14/14, discovery 38/38); false alarms 0/10 → 13/96, all on real listings (13 of the 14 real listings that passed the prefilter); no real listing reported to a platform (low scorer confidence downgraded every one to escalate). Causes: the ORB own-photo route alone (16–42 inliers vs threshold 16) labels real studio bottle shots `copied_photos`; CLIP ≥ `lookalike_clip` plus "no brand" labels generic bottles `lookalike`. Not retuned yet: any fix should be tuned on the dev split's real listings only.
- Real-noise fixes (two rules, one new value each, tuned on the dev split incl. its 129 real listings with `python -m eval.tune_own_photo --labels seed/labels_real.json`; the original thresholds are reproduced unchanged):
  - ORB own-photo route needs CLIP corroboration on the same image: `own_photo.orb_clip_min` 0.85 (dev: 47 real images passed ORB alone, max CLIP 0.84; all 91 gold-derived images still matched). `0` restores ORB alone.
  - Lookalike element `inspired_kw_or_no_mark` → `inspired_kw_or_close_copy`: copying wording, or no mark AND CLIP ≥ `thresholds.lookalike_silent_clip` 0.833. This deviates from §7.3's "(inspired_kw OR no brand)": resemblance alone is not evidence of passing off. Dev is not separable (one silent lookalike at 0.8339, a real listing at 0.85), so the value keeps every dev lookalike (rounded down). `null` restores the original rule.
  - Result (run 6 vs run 4, test split, scorer not refitted): false alarms on real listings 13/86 → 3/86; report precision 15/15; escalate recall 14/14 → 13/14 (a silent lookalike at CLIP 0.80 is no longer escalated); report recall 16/18 → 15/18 (a counterfeit whose photo matched only by ORB on the printed mark lost the own-photo feature, so scorer confidence fell 0.80 → 0.54 and it was escalated rather than reported). Dev: synthetic metrics unchanged, false alarms 22 → 6.
  - Not done yet: the scorer was fitted with the old own-photo definition; refitting it on the dev split (`python -m eval.fit_scorer`) would bring its inputs in line with the new rule.
- Phase 9 UI (`make ui` → `streamlit run ui/app.py`): Streamlit only reads the DB and writes decisions through `bh.enforce.approvals` / `bh.trace.resume_task`. Every pipeline process (pipeline, submit loop, mock sites, drafting, eval) is spawned as a separate OS process by `ui/workers.py` (pid + log in `data/ui/`). The approver's name comes from the sidebar "Reviewing as" field and is required for Approve and submit / Reject.
- The staff Run page launches the LangGraph orchestrator (`bh.orchestrator.run --brand <slug> [--llm]`). `ui/pipeline.py` (a chain of the stage CLIs) is only a fallback for checkouts without `bh/orchestrator/run.py`.
- The Evaluation page renders `data/eval/report.json` (schema in `ui/views/evaluation.py`) and computes nothing itself. Phase 10's `eval/evaluate.py` must write that file.
- **Product direction change (shopper-facing UI).** The end users are now shoppers, not the brand's lawyers. `ui/app.py` is a shopper app: give a product name and/or a photo, watch the agent work (current step, which site it is on and the page it has open, a plain-language log), then see listings grouped as "Looks genuine" / "Genuine, but check first" / "Avoid" / "Similar products from other makers", each with one plain sentence of why. It shows no scores, statutes, tokens or timings. The lawyer UI (all six SPEC §8 pages) moves to `ui/staff.py` (`make staff`), unchanged.
- Shopper searches run through `bh/shopper.py`: load brand → identify the product in the photo (CLIP vs gold photos, `shopper.photo_match_min`) → up to `shopper.max_queries` queries (the shopper's words, the identified product, the mark) → text search on every marketplace + search by photo where the adapter supports it (`TradeLoftAdapter.visual_search`) → prefilter → capture (≤ `shopper.max_listings`) → classify with the same `rules.yaml`. No LLM. Shopper searches write to their own DB (`shopper.db_url`) so they never enter the enforcement review queue. If the calibrated scorer hasn't been fitted, the rule outcome alone decides the shopper verdict.
- Shoppers can report a listing in the "Avoid" group to the brand's legal team (`ui/shop/report.py`). The report copies the listing, features and classification from the shopper DB into the enforcement DB as a case with `shopper_reports` ≥ 1 and runs `decide` on it. When the rule has a portal-reportable right (trade mark or copyright) the case goes to `awaiting_approval` even if its confidence would normally downgrade it to escalate, because a person asked for it to be reviewed; the reasoning says so. Nothing is filed with a marketplace until a named lawyer approves it in `ui/staff.py`; the shopper never signs declarations (§1). The shopper's card shows the report's progress: reported, approved, taken down, or not acted on. Repeat reports increment the count rather than creating duplicate cases.
- The shopper UI copy is brand-neutral (no brand or product category is named in the home page, feed or summary). Brand names still appear where they describe a specific result, e.g. "Uses the Vaelo name, but shows signs of a fake".
- **Shopper reports go to the marketplace.** Each marketplace now has its own "Report this listing" complaint form (`mocksites/listing_reports.py`; a link on every listing page; the submitted report has its own page at `/<site>/reports/<ref>` showing every input and the evidence). From an "Avoid" result the shopper opens a report dialog, prefilled from the verdict, and confirms reason, description, name, email and the buyer's declaration. The agent (`bh/enforce/listing_report.py`, one process per report) then opens the listing in a browser, follows its report link, fills in and submits the form, and records the reference and report URL on a `ShopperReport` row (shopper DB). The shopper app's sidebar ("Your reports") lists every report with live progress and a link that opens the submitted form in a new tab. The shopper signs only a buyer's declaration ("accurate to the best of my knowledge"), never the rights-owner declaration, which remains lawyer-only in the IP portal. Reporting to the brand's legal team (above) is now an optional checkbox in the same dialog.
- Demo notices removed at the team's request: the shopper app, the marketplace pages, the mock-sites index, the portal header and the staff app's banner no longer say the data is a demo, synthetic or fictional. This overrides §6's "clearly labelled as such in the UI" for the trade mark registration. The data itself is still entirely fictional (§1), and the staff banner still says "Decision support for a lawyer — not legal advice".
- UI wired to the merged pipeline. Shopper searches (`bh/shopper.py`) use the pipeline's LLM features when `shopper.use_llm` allows (auto: an API key is set): the extractor fallback, the borderline vision check and the navigator. Adapter-less sites are searched by replaying their learned recipe with no LLM, or by the navigator when an LLM is available (`shopper.navigator_sites`). The shopper feed narrates the navigator's decisions (action, target, reason), cookie answers, recipe replays and recipe learning. The staff Run page passes the orchestrator's `--llm`, `--navigator`, `--until`, `--sites` and `--brand-dir` (validated with `validate_brand_dir` before Start is enabled) and shows the run's stages from its tasks. The Trace page shows each navigator decision and replayed recipe step.
