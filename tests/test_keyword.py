import sqlite3
from pathlib import Path

from search import keyword_search, parse_query, search_terms


ROOT = Path(__file__).parents[1]


def test_parse_query_price_patterns() -> None:
    assert parse_query("under $30") == ("", 30.0)
    assert parse_query("below 50") == ("", 50.0)
    assert parse_query("less than $20 for mugs") == ("for mugs", 20.0)
    assert parse_query("running shoes <= 40") == ("running shoes", 40.0)


def test_injection_text_returns_normally_without_mutation() -> None:
    with sqlite3.connect(ROOT / "products.db") as conn:
        before = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        results = keyword_search(conn, "x' OR 1=1 --", None, None, False)
        after = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert isinstance(results, list)
    assert before == after == 600


def test_empty_query_is_handled() -> None:
    with sqlite3.connect(ROOT / "products.db") as conn:
        assert keyword_search(conn, "", None, None, False) == []


def test_possessive_suffix_does_not_become_an_fts_term() -> None:
    assert search_terms("youngster's corner") == ["youngster", "corner"]
