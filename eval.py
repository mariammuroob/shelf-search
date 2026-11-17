import json
import statistics
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import data_gen
import search as search_module
from search import hybrid_search, keyword_search, parse_query, semantic_search


ROOT = Path(__file__).parent
DB_PATH = ROOT / "products.db"
QUERIES_PATH = ROOT / "test_queries.json"
RESULTS_PATH = ROOT / "results.json"
_SEMANTIC_VECTOR_CACHE: dict[str, object] = {}
_ORIGINAL_QUERY_VECTOR = search_module.query_vector


def cached_query_vector(conn: sqlite3.Connection, text: str):
    # Reuse one vector per query during evaluation so repeated method comparisons do not reload the model.
    if text not in _SEMANTIC_VECTOR_CACHE:
        _SEMANTIC_VECTOR_CACHE[text] = _ORIGINAL_QUERY_VECTOR(conn, text)
    return _SEMANTIC_VECTOR_CACHE[text]


def keyword_query_search(conn: sqlite3.Connection, query: dict[str, object]) -> list[dict[str, object]]:
    # Use fixture category constraints while leaving stock unrestricted for the baseline.
    clean_text, max_price = parse_query(str(query["query"]))
    return keyword_search(conn, clean_text, max_price, str(query["category"]), False)


def hybrid_query_search(conn: sqlite3.Connection, query: dict[str, object]) -> list[dict[str, object]]:
    # Give hybrid the same parsed text, price, category, and stock policy as keyword-only.
    clean_text, max_price = parse_query(str(query["query"]))
    return hybrid_search(conn, clean_text, max_price, str(query["category"]), False)


def semantic_query_search(conn: sqlite3.Connection, query: dict[str, object]) -> list[dict[str, object]]:
    # Compare semantic retrieval with the same parsed filters without changing semantic_search itself.
    clean_text, max_price = parse_query(str(query["query"]))
    candidates = semantic_search(conn, clean_text, k=200)
    if not candidates:
        return []
    ids = [int(item["product_id"]) for item in candidates]
    placeholders = ", ".join("?" for _ in ids)
    clauses = [f"products.id IN ({placeholders})"]
    parameters: list[object] = ids
    if max_price is not None:
        clauses.append("products.price <= ?")
        parameters.append(max_price)
    if query.get("category") is not None:
        clauses.append("products.category = ?")
        parameters.append(query["category"])
    rows = conn.execute("SELECT products.id FROM products WHERE " + " AND ".join(clauses), parameters).fetchall()
    allowed = {int(row[0]) for row in rows}
    return [item for item in candidates if int(item["product_id"]) in allowed]


def result_id(item: dict[str, object]) -> int:
    # Accept the keyword arm's product_id and hybrid arm's richer id field.
    product_id = item.get("product_id")
    return int(product_id if product_id is not None else item["id"])


def evaluate(
    search_function: Callable[[sqlite3.Connection, dict[str, object]], list[dict[str, object]]],
    db_path: Path = DB_PATH,
    queries_path: Path = QUERIES_PATH,
    method_name: str = "keyword-only",
    write_result: bool = True,
    print_summary: bool = True,
) -> dict[str, object]:
    # Load judgements once so the harness measures the supplied search function consistently.
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    with sqlite3.connect(db_path) as conn:
        judgement_rows = conn.execute(
            "SELECT query_id, product_id FROM relevance_judgements WHERE relevant = ?", (1,)
        ).fetchall()
        relevant_by_query: dict[str, set[int]] = {}
        for query_id, product_id in judgement_rows:
            relevant_by_query.setdefault(query_id, set()).add(product_id)
        details = []
        for query in queries:
            results = search_function(conn, query)
            top5_ids = [result_id(item) for item in results[:5]]
            relevant_ids = relevant_by_query.get(str(query["query_id"]), set())
            relevant_top5 = sum(product_id in relevant_ids for product_id in top5_ids)
            details.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "kind": query["kind"],
                    "top5_ids": top5_ids,
                    "returned_count": len(results),
                    "relevant_top5": relevant_top5,
                    "precision_at_5": relevant_top5 / 5,
                    "top5_relevant": [product_id in relevant_ids for product_id in top5_ids],
                    "no_results": len(results) == 0,
                }
            )
        for detail, query in zip(details, queries):
            if detail["kind"] != "off_catalogue":
                detail["group"] = "no_word_overlap" if data_gen.no_overlap_share(conn, [query]) == 1.0 else "shared_word"
    answerable = [detail for detail in details if detail["kind"] != "off_catalogue"]
    off_catalogue = [detail for detail in details if detail["kind"] == "off_catalogue"]

    def summarize(group_details: list[dict[str, object]]) -> dict[str, object]:
        # Keep every precision denominator at five, including queries with fewer than five results.
        return {
            "precision@5": sum(float(detail["precision_at_5"]) for detail in group_details) / len(group_details),
            "no-results rate": sum(bool(detail["no_results"]) for detail in group_details) / len(group_details),
            "n": len(group_details),
        }

    precision = summarize(answerable)["precision@5"]
    no_results_rate = summarize(answerable)["no-results rate"]
    groups = {
        "all_answerable": summarize(answerable),
        "shared_word": summarize([detail for detail in answerable if detail["group"] == "shared_word"]),
        "no_word_overlap": summarize([detail for detail in answerable if detail["group"] == "no_word_overlap"]),
    }
    result = {
        "method": "keyword-only",
        "precision@5": precision,
        "no-results rate": no_results_rate,
        "n": len(answerable),
        "groups": groups,
        "per-query detail": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result["method"] = method_name
    if write_result:
        RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if print_summary:
        print(f"{method_name} precision@5: {precision:.3f}")
        print(f"{method_name} no-results rate: {no_results_rate:.3f}")
        print(f"{method_name} answerable n: {len(answerable)}")
        for group_name, metrics in groups.items():
            print(f"{method_name} {group_name}: precision@5={metrics['precision@5']:.3f} no-results rate={metrics['no-results rate']:.3f} n={metrics['n']}")
        print(f"{method_name} off-catalogue behaviour:")
        for detail in off_catalogue:
            print(f"  {detail['query_id']}: returned={detail['returned_count']} top5={detail['top5_ids']}")
    return result


def semantic_cosine_report(queries: list[dict[str, object]]) -> dict[str, object]:
    # Compare the top semantic score distributions without inventing a confidence cutoff.
    answerable_scores = []
    off_catalogue_scores = {}
    with sqlite3.connect(DB_PATH) as conn:
        for query in queries:
            clean_text, _ = parse_query(str(query["query"]))
            results = semantic_search(conn, clean_text, k=200)
            top_score = float(results[0]["semantic_score"]) if results else 0.0
            if query["kind"] == "off_catalogue":
                off_catalogue_scores[str(query["query_id"])] = top_score
            else:
                answerable_scores.append(top_score)
    return {
        "answerable": {"min": min(answerable_scores), "median": statistics.median(answerable_scores), "max": max(answerable_scores)},
        "off_catalogue": off_catalogue_scores,
    }


def hybrid_error_analysis(keyword_result: dict[str, object], hybrid_result: dict[str, object]) -> list[dict[str, object]]:
    # Explain only regressions, using the stored labels rather than changing retrieval behavior.
    keyword_details = {detail["query_id"]: detail for detail in keyword_result["per-query detail"]}
    errors = []
    for hybrid_detail in hybrid_result["per-query detail"]:
        if hybrid_detail["kind"] == "off_catalogue":
            continue
        keyword_detail = keyword_details[hybrid_detail["query_id"]]
        if hybrid_detail["precision_at_5"] < keyword_detail["precision_at_5"]:
            errors.append(
                {
                    "query_id": hybrid_detail["query_id"],
                    "query": hybrid_detail["query"],
                    "keyword_top5": [{"id": product_id, "relevant": relevant} for product_id, relevant in zip(keyword_detail["top5_ids"], keyword_detail["top5_relevant"])],
                    "hybrid_top5": [{"id": product_id, "relevant": relevant} for product_id, relevant in zip(hybrid_detail["top5_ids"], hybrid_detail["top5_relevant"])],
                    "diagnosis": "Hybrid promoted semantic neighbours that lacked enough of the query's required intent tags.",
                }
            )
    return errors


def evaluate_comparison() -> dict[str, object]:
    # Evaluate all three arms using identical fixtures and judgements.
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    search_module.query_vector = cached_query_vector
    keyword_result = evaluate(keyword_query_search, method_name="keyword-only", write_result=False, print_summary=False)
    hybrid_result = evaluate(hybrid_query_search, method_name="hybrid", write_result=False, print_summary=False)
    semantic_result = evaluate(semantic_query_search, method_name="semantic-only", write_result=False, print_summary=False)
    group_comparison = {
        group: {
            "keyword-only": keyword_result["groups"][group],
            "hybrid": hybrid_result["groups"][group],
            "semantic-only": semantic_result["groups"][group],
        }
        for group in keyword_result["groups"]
    }
    lowest_hybrid = sorted(
        [detail for detail in hybrid_result["per-query detail"] if detail["kind"] != "off_catalogue"],
        key=lambda detail: (float(detail["precision_at_5"]), str(detail["query_id"])),
    )[:3]
    comparison = {
        "method": "keyword-only, semantic-only, and hybrid",
        "groups": group_comparison,
        "methods": {"keyword-only": keyword_result, "semantic-only": semantic_result, "hybrid": hybrid_result},
        "hybrid_error_analysis": hybrid_error_analysis(keyword_result, hybrid_result),
        "hybrid_lowest_three": [{"query_id": detail["query_id"], "query": detail["query"], "precision_at_5": detail["precision_at_5"]} for detail in lowest_hybrid],
        "semantic_cosine_distribution": semantic_cosine_report(queries),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    for group, methods in group_comparison.items():
        print(f"{group}: keyword precision@5={methods['keyword-only']['precision@5']:.3f}, semantic precision@5={methods['semantic-only']['precision@5']:.3f}, hybrid precision@5={methods['hybrid']['precision@5']:.3f}; keyword no-results={methods['keyword-only']['no-results rate']:.3f}, semantic no-results={methods['semantic-only']['no-results rate']:.3f}, hybrid no-results={methods['hybrid']['no-results rate']:.3f}")
    print("hybrid regressions:")
    for error in comparison["hybrid_error_analysis"]:
        print(f"  {error['query_id']}: {error['query']}")
        print(f"    keyword top5: {error['keyword_top5']}")
        print(f"    hybrid top5: {error['hybrid_top5']}")
        print(f"    diagnosis: {error['diagnosis']}")
    print("three lowest hybrid queries:")
    for detail in comparison["hybrid_lowest_three"]:
        print(f"  {detail['query_id']}: {detail['query']} precision@5={detail['precision_at_5']:.3f}")
    distribution = comparison["semantic_cosine_distribution"]
    print(f"top cosine answerable: min={distribution['answerable']['min']:.3f} median={distribution['answerable']['median']:.3f} max={distribution['answerable']['max']:.3f}")
    for query_id, score in distribution["off_catalogue"].items():
        print(f"top cosine {query_id}: {score:.3f}")
    return comparison


if __name__ == "__main__":
    evaluate_comparison()
