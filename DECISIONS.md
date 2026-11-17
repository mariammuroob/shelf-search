# Decisions

## Phase 5: API and synthetic search page

- Decision: load the active encoder and normalized embeddings matrix once in FastAPI lifespan, serve the dev-selected weight 0.25, and log searches with a fail-soft SQLite write. Show real relaxed-filter and outside-filter suggestions only when a method is empty.
- Alternatives considered: loading the model on every request or treating nearest neighbours as proof of relevance.
- Reason: startup caching keeps requests responsive and measured startup working set was 218.2 MB (252.2 MB peak) on local Windows, below the 512 MB Render free-tier limit. The API reports no confidence flag because the off-catalogue cosine scores overlap answerable scores.
- Deployment check: build_data.py runs data_gen.py then embed.py with checked subprocesses and does not invoke phase4b.py test. results.json is tracked in Git and served byte-for-byte by /api/eval. The results were measured locally on Windows; a Linux rebuild on Render may differ slightly in embeddings.
- Files changed: api.py, static/index.html, tests/test_guardrails.py, DECISIONS.md.
- Commands to run: `python -m pytest tests --confcutdir=.` and `uvicorn api:app --host 127.0.0.1 --port 8000` after installing requirements.

## Phase 1: repository skeleton

- Decision: keep the API to a health route and a static placeholder, with data generation as a print-only stub.
- Alternatives considered: begin catalogue, search, or evaluation implementation now.
- Reason: this phase explicitly limits scope to the deployable skeleton.
- Files changed: AGENTS.md, .gitignore, DECISIONS.md, README.md, api.py, build_data.py, render.yaml, requirements.txt, static/index.html, test_queries.json, tests/.gitkeep.
- Commands to run: `python -m venv .venv`; `.venv\\Scripts\\pip install -r requirements.txt`; `.venv\\Scripts\\python build_data.py`; `.venv\\Scripts\\python -m pytest tests --confcutdir=.`; `.venv\\Scripts\\python -m uvicorn api:app --host 0.0.0.0 --port 8000`.

## Phase 1: seeded synthetic data

- Decision: generate 600 deterministic products with balanced category/tag bundles, explicit all-product relevance judgements, and 30 JSON query fixtures.
- Alternatives considered: hand-authored rows, random tags without coverage checks, or judgements only for category matches.
- Reason: seeded templates keep the catalogue reproducible, ensure every answerable fixture has at least five matches, and make zero judgements explicit for non-matches.
- Files changed: data_gen.py, test_queries.json, products.db, DECISIONS.md.
- Commands to run: `python data_gen.py`; `python -m pytest tests --confcutdir=.`; `uvicorn api:app --host 0.0.0.0 --port 8000`.

## Phase 2: keyword baseline and evaluation harness

- Decision: use a bound-parameter FTS5 OR query with optional bound filters, and make evaluation accept the search function as an argument.
- Alternatives considered: concatenate MATCH/filter input, use plain AND, or hard-code evaluation to one search implementation.
- Reason: OR preserves a fair natural-language keyword baseline, parameter binding protects the database, and an injectable function supports later hybrid comparison.
- Files changed: search.py, eval.py, build_data.py, tests/test_keyword.py, results.json, DECISIONS.md.
- Commands to run: `python build_data.py`; `python -m pytest tests --confcutdir=.`; `python eval.py`; `uvicorn api:app --host 0.0.0.0 --port 8000`.

## Fairness check: possessive-token fix

- Decision: add the possessive suffix `s` to the keyword stopwords so text such as `youngster's` cannot send a stray one-letter FTS term.
- Alternatives considered: leave the token as-is or rewrite query text before parsing.
- Reason: the fairness report exposed this as a real sanitisation bug; price words and planned category/stock filters were already correct.
- Files changed: search.py, tests/test_keyword.py, DECISIONS.md.
- Commands to run: `python -m pytest tests --confcutdir=.`; `python eval.py`.

## Phase 2 fairness fix: freeze keyword arm

- Decision: recreate FTS5 with `porter unicode61`, add only generic filler stopwords, and report all/shared-word/paraphrase groups before freezing the keyword arm.
- Alternatives considered: tune stopwords or tokenizer behavior per query, or leave the baseline unchanged before hybrid work.
- Reason: stemming improves ordinary variants such as gift/gifts and throw/throws; strengthening the baseline before any hybrid result exists prevents accusations that the hybrid was favored.
- Files changed: data_gen.py, search.py, eval.py, results.json, README.md, DECISIONS.md, products.db.
- Commands to run: `python build_data.py`; `python -m pytest tests --confcutdir=.`; `python eval.py`; `uvicorn api:app --host 0.0.0.0 --port 8000`.
- Freeze: no further changes to the keyword search, parser, stopwords, or tokenizer while hybrid search is built.

## Phase 3: embeddings and hybrid search

- Decision: use verified fastembed 0.9.1 `TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./models")`, with persisted float32 vectors and a TF-IDF fallback; fuse 200-result keyword and semantic lists with deterministic RRF k=60 before bound SQL filters.
- Alternatives considered: require the model with no fallback, filter each arm before fusion, or apply hybrid changes to the frozen keyword arm.
- Reason: the local fallback keeps builds usable, wide-list fusion avoids filter starvation, and preserving the keyword arm keeps the comparison fair.
- Low-confidence reporting uses a fixed top-cosine threshold of 0.35 for off-catalogue queries.
- Files changed: embed.py, search.py, eval.py, build_data.py, tests/test_hybrid.py, results.json, DECISIONS.md, products.db, models/.
- Commands to run: `python build_data.py`; `python -m pytest tests --confcutdir=.`; `python eval.py`; `uvicorn api:app --host 0.0.0.0 --port 8000`.
- Freeze: the keyword search, parser, stopwords, and tokenizer are now frozen while hybrid work continues.

## Phase 4: three-method evaluation and confidence review

- Decision: add semantic-only metrics, hybrid-vs-keyword error analysis, and top-cosine distributions to the evaluation report without changing retrieval code or tuning any values.
- Alternatives considered: tune hybrid parameters, regroup q09, or ship a low-confidence flag based on the cosine scores.
- Reason: hybrid is lower overall and on shared-word queries; q09 is effectively a paraphrase despite its assigned group; q29/q30 are not separated by cosine because q30 falls inside the answerable range, so no low-confidence flag is planned for the API or UI.
- Files changed: eval.py, README.md, results.json, DECISIONS.md.
- Commands to run: `python -m pytest tests --confcutdir=.`; `python eval.py`.
- Freeze: do not change search.py, the keyword arm, semantic arm, fusion, k, candidate size, stopwords, labels, or queries.

## Phase 4b: dev-selected weighted fusion

- Decision: create 24 separate dev queries with tag-rule labels (6 paraphrases), select `w=0.25` on dev precision@5, and commit that weight for the future API and UI while showing equal and weighted hybrid numbers.
- Alternatives considered: weights 0.5, 0.75, 1.0; dev precision@5 was respectively 0.275, 0.250, 0.258 versus 0.308 at 0.25. The best exceeded 1.0 by 0.050, above the precommitted 0.02 margin.
- Reason: choose the weight before held-out test evaluation and preserve all frozen retrieval arms. The dev set derives from the same generator and tag rules, so it is not independent evidence.
- Held-out test: weighted precision@5 0.371 overall, 0.419 shared-word, 0.229 paraphrase; no-results rate 0 in each. The first held-out invocation failed on an off-catalogue group reporting lookup before writing metrics; the same locked weight was rerun after fixing only that lookup.
- Files changed: dev_gen.py, dev_queries.json, phase4b.py, dev_selection.json, results.json, README.md, DECISIONS.md.
- Commands to run: `python dev_gen.py`; `python phase4b.py dev`; `python phase4b.py test` (test is guarded against rerun); `python -m pytest tests --confcutdir=.`; `uvicorn api:app --host 127.0.0.1 --port 8000`.

## Phase 1 fix 2: title variants and label limitation

- Decision: add colour, size, material, and pack-count variants to generated titles while keeping tags, judgements, and query wording unchanged.
- Alternatives considered: leave the 185 distinct titles, or alter tag assignment and relevance rules to make titles appear more varied.
- Reason: natural title variants raise title distinctness without changing the intent-label experiment.
- Every query's relevant count is a multiple of five because each category has 75 products built from five repeating tag-template groups, each contributing 15 products. Yes, the pattern comes from tags being assigned per template group, so a query selects whole groups (or overlaps between groups) rather than independently varied labels. This is a limitation: labels derive from the generator tags, so the evaluation partly measures my own generator.
- Files changed: data_gen.py, README.md, DECISIONS.md, products.db.
- Commands to run: `python data_gen.py`; `python -m pytest tests --confcutdir=.`; `uvicorn api:app --host 0.0.0.0 --port 8000`.

## Phase 1 fix: variation and subjective queries

- Decision: vary descriptions with synonym pools, three sentence structures, material/use-case attributes, and style identifiers; rewrite seven answerable queries with subjective paraphrases that have no text-word overlap with their relevant products.
- Alternatives considered: accept the original 578 shared descriptions and rely on keyword overlap, or alter relevance to depend on query wording.
- Reason: descriptions should be near-duplicate exceptions, while about 25% subjective searches mirrors Target's reported share; relevance remains defined by intent tags and category, not wording.
- Files changed: data_gen.py, test_queries.json, products.db, DECISIONS.md.
- Commands to run: `python data_gen.py`; `python -m pytest tests --confcutdir=.`; `uvicorn api:app --host 0.0.0.0 --port 8000`.
