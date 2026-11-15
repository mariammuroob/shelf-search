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
