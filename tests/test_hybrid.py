import sqlite3
from pathlib import Path

from search import hybrid_search, rrf_fuse


ROOT = Path(__file__).parents[1]


def test_rrf_fusion_arithmetic_and_single_arm_product() -> None:
    fused = rrf_fuse([1, 2], [2, 3], k=60)
    by_id = {item["id"]: item for item in fused}
    assert by_id[1]["rrf_score"] == 1 / 61
    assert by_id[2]["rrf_score"] == 1 / 61 + 1 / 62
    assert by_id[3]["rrf_score"] == 1 / 62
    assert by_id[1]["semantic_rank"] is None


def test_tight_price_filter_still_returns_existing_matches() -> None:
    with sqlite3.connect(ROOT / "products.db") as conn:
        results = hybrid_search(conn, "sturdy organisers", 30.0, "home", False)
    assert results
    assert all(result["price"] <= 30.0 for result in results)


def test_hybrid_same_inputs_are_deterministic() -> None:
    with sqlite3.connect(ROOT / "products.db") as conn:
        first = hybrid_search(conn, "sturdy organisers", 30.0, "home", False)
        second = hybrid_search(conn, "sturdy organisers", 30.0, "home", False)
    assert first == second
