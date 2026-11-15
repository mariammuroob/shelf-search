# Decisions

## Phase 1: repository skeleton

- Decision: keep the API to a health route and a static placeholder, with data generation as a print-only stub.
- Alternatives considered: begin catalogue, search, or evaluation implementation now.
- Reason: this phase explicitly limits scope to the deployable skeleton.
- Files changed: AGENTS.md, .gitignore, DECISIONS.md, README.md, api.py, build_data.py, render.yaml, requirements.txt, static/index.html, test_queries.json, tests/.gitkeep.
- Commands to run: `python -m venv .venv`; `.venv\\Scripts\\pip install -r requirements.txt`; `.venv\\Scripts\\python build_data.py`; `.venv\\Scripts\\python -m pytest tests --confcutdir=.`; `.venv\\Scripts\\python -m uvicorn api:app --host 0.0.0.0 --port 8000`.
