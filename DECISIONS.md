# Decisions

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
