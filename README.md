# What this is

Shelf-search is a synthetic small-retail hybrid product-search demo. It compares SQLite FTS5 keyword retrieval, local semantic retrieval, and Reciprocal Rank Fusion on a generated catalogue.

# How it was built

The catalogue contains 600 generated products. The keyword arm uses SQLite FTS5 with Porter stemming and bound filters; the semantic arm uses local `BAAI/bge-small-en-v1.5` embeddings through fastembed 0.9.1, with a TF-IDF fallback. Hybrid retrieval fuses the top 200 candidates from each arm with RRF k=60 before applying filters.

# Target references

Target references are contextual only. No Target branding, logo, scraping, or real Target data is used.

# Synthetic data

The catalogue, queries, hidden intent tags, and relevance labels are synthetic and generated locally with seed 42.

# What is NOT claimed

- No AlloyDB or ScaNN parity.
- No production scale.
- No Target data.
- No Target metrics presented as this project's result.
- No personalised pricing.
- No replica of Target's system.

# Results

Evaluation timestamp: 2026-11-17. Embedding backend: fastembed 0.9.1, `BAAI/bge-small-en-v1.5`.

| Group | n | Keyword precision@5 | Semantic precision@5 | Hybrid precision@5 | Keyword no-results | Semantic no-results | Hybrid no-results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All answerable | 28 | 0.314 | 0.150 | 0.293 | 0.214 | 0.000 | 0.000 |
| Shared-word | 21 | 0.419 | 0.124 | 0.314 | 0.048 | 0.000 | 0.000 |
| Paraphrase | 7 | 0.000 | 0.229 | 0.229 | 0.714 | 0.000 | 0.000 |

Hybrid did not beat keyword-only on overall precision@5 (0.293 vs 0.314) and was worse on shared-word queries (0.314 vs 0.419). It was better on paraphrase queries (0.229 vs 0.000) and returned results for every query, which is partly by construction because semantic search always returns neighbours.

The catalogue, queries, and labels are synthetic and generated from the same intent tags, so this evaluation partly measures my own generator. Seven of 28 queries are paraphrases by design. The result is directional on a small set. Target's reported figures are theirs, not mine.

## What I would try next

Weighted fusion or query routing, tuned on a separate held-out development query set, never on these 30 queries.

# Limits

Every query's relevant count is a multiple of five because each category has 75 products built from five repeating tag-template groups, each contributing 15 products. Yes, the pattern comes from tags being assigned per template group, so a query selects whole groups (or overlaps between groups) rather than independently varied labels. This is a limitation: labels derive from the generator tags, so the evaluation partly measures my own generator.

Seven of the 28 answerable queries are paraphrases by design.

Grouping caveat: q09 returns zero keyword results on its real terms (`little OR occupied OR trip`) and is effectively a paraphrase query, although it remains in the shared-word group for the headline numbers.

The top-cosine distribution does not support a useful low-confidence flag: answerable queries range from 0.586 to 0.774, q29 is 0.551, and q30 is 0.659. No low-confidence flag is planned for the API or UI.

# How to run

```text
python build_data.py
python -m pytest tests --confcutdir=.
python eval.py
uvicorn api:app --host 0.0.0.0 --port 8000
```
