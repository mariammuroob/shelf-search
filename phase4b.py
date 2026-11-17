import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import search
from search import keyword_search, parse_query, semantic_search


ROOT = Path(__file__).parent
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
SELECTED_PATH = ROOT / "dev_selection.json"
_vectors = {}
_original_vector = search.query_vector


def cached_vector(conn, text):
    # Cache identical query vectors during a reporting run to avoid repeated model loading.
    if text not in _vectors:
        _vectors[text] = _original_vector(conn, text)
    return _vectors[text]


def weighted_results(conn, query, weight):
    # Keep both frozen candidate arms and apply the same bound filters after weighted RRF.
    text, price = parse_query(query["query"])
    keyword = keyword_search(conn, text, None, None, False, k=200)
    semantic = semantic_search(conn, text, k=200)
    scores = {}
    for rank, item in enumerate(keyword, 1):
        scores[int(item["product_id"])] = 1 / (60 + rank)
    for rank, item in enumerate(semantic, 1):
        product_id = int(item["product_id"])
        scores[product_id] = scores.get(product_id, 0) + weight / (60 + rank)
    if not scores:
        return []
    ids = list(scores)
    placeholders = ", ".join("?" for _ in ids)
    clauses = [f"id IN ({placeholders})", "category = ?"]
    parameters = ids + [query["category"]]
    if price is not None:
        clauses.append("price <= ?")
        parameters.append(price)
    allowed = {row[0] for row in conn.execute("SELECT id FROM products WHERE " + " AND ".join(clauses), parameters)}
    return sorted((product_id for product_id in ids if product_id in allowed), key=lambda product_id: (-scores[product_id], product_id))[:5]


def dev():
    # Select on dev alone and persist the decision before reading held-out results.
    queries = json.loads((ROOT / "dev_queries.json").read_text(encoding="utf-8"))
    search.query_vector = cached_vector
    with sqlite3.connect(ROOT / "products.db") as conn:
        grid = {}
        for weight in WEIGHTS:
            grid[str(weight)] = sum(sum(product_id in set(query["relevant_ids"]) for product_id in weighted_results(conn, query, weight)) / 5 for query in queries) / len(queries)
    best = min(WEIGHTS, key=lambda weight: (-grid[str(weight)], weight))
    selected = best if grid[str(best)] - grid["1.0"] >= 0.02 else 1.0
    output = {"grid": grid, "selected_weight": selected, "n": len(queries), "paraphrase_n": sum(q["group"] == "no_word_overlap" for q in queries), "timestamp": datetime.now(timezone.utc).isoformat()}
    SELECTED_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("dev grid:", grid, "selected:", selected)


def test_once():
    # Run only the selected weight on held-out queries; reuse the already frozen three-method report.
    selection = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    weight = float(selection["selected_weight"])
    queries = json.loads((ROOT / "test_queries.json").read_text(encoding="utf-8"))
    report_path = ROOT / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "hybrid_weighted" not in report["methods"], "held-out weighted test already run"
    search.query_vector = cached_vector
    with sqlite3.connect(ROOT / "products.db") as conn:
        labels = {}
        for query_id, product_id in conn.execute("SELECT query_id, product_id FROM relevance_judgements WHERE relevant = ?", (1,)):
            labels.setdefault(query_id, set()).add(product_id)
        details = []
        for query in queries:
            ids = weighted_results(conn, query, weight)
            relevant = labels.get(query["query_id"], set())
            details.append({"query_id": query["query_id"], "query": query["query"], "kind": query["kind"], "group": next((d.get("group") for d in report["methods"]["keyword-only"]["per-query detail"] if d["query_id"] == query["query_id"]), None), "top5_ids": ids, "top5_relevant": [product_id in relevant for product_id in ids], "precision_at_5": sum(product_id in relevant for product_id in ids) / 5, "no_results": not ids})
    groups = {}
    for name, subset in (("all_answerable", [d for d in details if d["kind"] != "off_catalogue"]), ("shared_word", [d for d in details if d["group"] == "shared_word"]), ("no_word_overlap", [d for d in details if d["group"] == "no_word_overlap"])):
        groups[name] = {"precision@5": sum(d["precision_at_5"] for d in subset) / len(subset), "no-results rate": sum(d["no_results"] for d in subset) / len(subset), "n": len(subset)}
        report["groups"][name]["hybrid_weighted"] = groups[name]
    report["methods"]["hybrid_weighted"] = {"method": "hybrid_weighted", "weight": weight, "groups": groups, "per-query detail": details, "timestamp": datetime.now(timezone.utc).isoformat()}
    report["dev_selection"] = selection
    report["embedding_backend"] = sqlite3.connect(ROOT / "products.db").execute("SELECT backend, model FROM embedding_backend LIMIT 1").fetchone()
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, methods in report["groups"].items():
        print(name, "n=", methods["hybrid_weighted"]["n"], " ".join(f"{method}: P@5={values['precision@5']:.3f} no-results={values['no-results rate']:.3f}" for method, values in methods.items()))


if __name__ == "__main__":
    {"dev": dev, "test": test_once}[sys.argv[1]]()
