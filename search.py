import re
import sqlite3


STOPWORDS = {
    "a", "an", "and", "are", "away", "anything", "for", "from", "idea", "ideas", "in", "keep", "make", "of", "on", "one", "or", "s", "something", "start", "the", "to", "with"
}


def parse_query(text: str) -> tuple[str, float | None]:
    # Remove price language before tokenisation so the number is a filter, not a search term.
    lowered = text.lower()
    price_pattern = re.compile(
        r"(?:under|below|less\s+than)\s*\$?\s*(\d+(?:\.\d+)?)|<=\s*\$?\s*(\d+(?:\.\d+)?)"
    )
    match = price_pattern.search(lowered)
    max_price = None
    if match:
        value = next(group for group in match.groups() if group is not None)
        max_price = float(value)
        lowered = price_pattern.sub(" ", lowered, count=1)
    clean_text = " ".join(lowered.split())[:200]
    return clean_text, max_price


def search_terms(text: str) -> list[str]:
    # Restrict FTS input to safe alphanumeric tokens and remove common noise words.
    terms = re.findall(r"[a-z0-9]+", text.lower())
    return [term for term in terms if term not in STOPWORDS]


def keyword_search(
    conn: sqlite3.Connection,
    text: str,
    max_price: float | None,
    category: str | None,
    in_stock_only: bool,
    k: int = 200,
) -> list[dict[str, object]]:
    # OR keeps the keyword baseline useful for natural queries; AND would be unfairly brittle.
    terms = search_terms(text)
    if not terms:
        return []
    match_text = " OR ".join(terms)
    clauses = ["product_fts MATCH ?"]
    parameters: list[object] = [match_text]
    if max_price is not None:
        clauses.append("products.price <= ?")
        parameters.append(max_price)
    if category is not None:
        clauses.append("products.category = ?")
        parameters.append(category)
    if in_stock_only:
        clauses.append("products.in_stock = ?")
        parameters.append(1)
    parameters.append(k)
    sql = (
        "SELECT products.id FROM product_fts "
        "JOIN products ON products.id = product_fts.rowid "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY bm25(product_fts) ASC, products.id ASC LIMIT ?"
    )
    rows = conn.execute(sql, parameters).fetchall()
    return [{"product_id": row[0], "matched_terms": terms} for row in rows]
