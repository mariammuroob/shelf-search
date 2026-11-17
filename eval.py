import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import data_gen
from search import keyword_search, parse_query


ROOT = Path(__file__).parent
DB_PATH = ROOT / "products.db"
QUERIES_PATH = ROOT / "test_queries.json"
RESULTS_PATH = ROOT / "results.json"


def keyword_query_search(conn: sqlite3.Connection, query: dict[str, object]) -> list[dict[str, object]]:
    # Use fixture category constraints while leaving stock unrestricted for the baseline.
    clean_text, max_price = parse_query(str(query["query"]))
    return keyword_search(conn, clean_text, max_price, str(query["category"]), False)


def evaluate(
    search_function: Callable[[sqlite3.Connection, dict[str, object]], list[dict[str, object]]],
    db_path: Path = DB_PATH,
    queries_path: Path = QUERIES_PATH,
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
            top5_ids = [int(item["product_id"]) for item in results[:5]]
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
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"keyword-only precision@5: {precision:.3f}")
    print(f"keyword-only no-results rate: {no_results_rate:.3f}")
    print(f"answerable n: {len(answerable)}")
    for group_name, metrics in groups.items():
        print(f"{group_name}: precision@5={metrics['precision@5']:.3f} no-results rate={metrics['no-results rate']:.3f} n={metrics['n']}")
    print("off-catalogue behaviour:")
    for detail in off_catalogue:
        print(f"  {detail['query_id']}: returned={detail['returned_count']} top5={detail['top5_ids']}")
    return result


if __name__ == "__main__":
    evaluate(keyword_query_search)
