# Project rules

- Project: small retail hybrid search (SQLite FTS5 keyword + local embeddings, fused with Reciprocal Rank Fusion k=60), synthetic catalogue, measured with precision@5 and no-results rate versus a keyword-only baseline.
- Stack: Python 3.11, FastAPI, SQLite, fastembed (BAAI/bge-small-en-v1.5) with automatic TF-IDF (scikit-learn) fallback, plain HTML/CSS/vanilla JS in static/. No frameworks, no build step.
- Flat layout: data_gen.py, embed.py, search.py, eval.py, build_data.py, api.py, static/, tests/, test_queries.json, README.md, DECISIONS.md, render.yaml, requirements.txt.
- Parameterised SQL only. Never concatenate user text into SQL, including IN lists (generate placeholders, bind values).
- No API keys, no paid services, no scraping, no Target branding, logo or real Target data. All data is synthetic and the UI and README say so.
- One fixed price per product. No per-user pricing anywhere.
- Never invent a product to fill a result page.
- Small functions, a comment explaining each design decision, no clever abstractions.
- After each phase: run the tests and the app yourself, fix failures, append a short entry to DECISIONS.md (decision, alternatives considered, reason), and list the files changed and the commands to run.
- Do only the current phase. Do not build ahead.
