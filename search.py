import re
import sqlite3
from pathlib import Path

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "away", "anything", "for", "from", "idea", "ideas", "in", "keep", "make", "of", "on", "one", "or", "s", "something", "start", "the", "to", "with"
}
LOW_CONFIDENCE_THRESHOLD = 0.35


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


def embedding_backend(conn: sqlite3.Connection) -> tuple[str, str]:
    # Read the backend recorded by embed.py instead of guessing from local files.
    row = conn.execute("SELECT backend, model FROM embedding_backend LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("embedding backend has not been built")
    return str(row[0]), str(row[1])


def query_vector(conn: sqlite3.Connection, text: str) -> np.ndarray:
    # Use the same active backend for query vectors as was used for catalogue vectors.
    backend, model_name = embedding_backend(conn)
    if backend == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_name, cache_dir=str(Path(__file__).parent / "models"))
        return np.asarray(list(model.query_embed(text))[0], dtype=np.float32)
    import joblib

    vectorizer = joblib.load(Path(__file__).parent / "models" / "tfidf_vectorizer.joblib")
    return vectorizer.transform([text]).toarray()[0].astype(np.float32)


def semantic_search(conn: sqlite3.Connection, text: str, k: int = 200) -> list[dict[str, object]]:
    # Brute-force cosine similarity is transparent and sufficient for this small synthetic catalogue.
    query = query_vector(conn, text)
    query_norm = float(np.linalg.norm(query))
    rows = conn.execute("SELECT product_id, embedding FROM product_embeddings ORDER BY product_id").fetchall()
    scored = []
    for product_id, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        denominator = query_norm * float(np.linalg.norm(vector))
        score = float(np.dot(query, vector) / denominator) if denominator else 0.0
        scored.append((int(product_id), score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [{"product_id": product_id, "semantic_score": score, "semantic_rank": rank} for rank, (product_id, score) in enumerate(scored[:k], start=1)]


def rrf_fuse(keyword_ids: list[int], semantic_ids: list[int], k: int = 60) -> list[dict[str, object]]:
    # Reciprocal Rank Fusion gives each arm independent rank evidence and uses id for stable ties.
    keyword_ranks = {product_id: rank for rank, product_id in enumerate(keyword_ids, start=1)}
    semantic_ranks = {product_id: rank for rank, product_id in enumerate(semantic_ids, start=1)}
    product_ids = set(keyword_ranks) | set(semantic_ranks)
    fused = []
    for product_id in product_ids:
        keyword_rank = keyword_ranks.get(product_id)
        semantic_rank = semantic_ranks.get(product_id)
        score = (1 / (k + keyword_rank) if keyword_rank else 0) + (1 / (k + semantic_rank) if semantic_rank else 0)
        fused.append({"id": product_id, "keyword_rank": keyword_rank, "semantic_rank": semantic_rank, "rrf_score": score})
    return sorted(fused, key=lambda item: (-float(item["rrf_score"]), int(item["id"])))


def hybrid_search(
    conn: sqlite3.Connection,
    text: str,
    max_price: float | None,
    category: str | None,
    in_stock_only: bool,
    limit: int = 5,
) -> list[dict[str, object]]:
    # Retrieve wide unfiltered lists, fuse them, then apply all user filters in one bound SQL query.
    keyword_results = keyword_search(conn, text, None, None, False, k=200)
    semantic_results = semantic_search(conn, text, k=200)
    keyword_ids = [int(item["product_id"]) for item in keyword_results]
    semantic_ids = [int(item["product_id"]) for item in semantic_results]
    fused = rrf_fuse(keyword_ids, semantic_ids, k=60)
    if not fused:
        return []
    ids = [int(item["id"]) for item in fused]
    placeholders = ", ".join("?" for _ in ids)
    clauses = [f"products.id IN ({placeholders})"]
    parameters: list[object] = ids
    if max_price is not None:
        clauses.append("products.price <= ?")
        parameters.append(max_price)
    if category is not None:
        clauses.append("products.category = ?")
        parameters.append(category)
    if in_stock_only:
        clauses.append("products.in_stock = ?")
        parameters.append(1)
    rows = conn.execute(
        "SELECT products.id, products.title, products.brand, products.category, products.price, products.rating, products.in_stock FROM products WHERE "
        + " AND ".join(clauses),
        parameters,
    ).fetchall()
    products = {int(row[0]): row for row in rows}
    keyword_by_id = {int(item["product_id"]): item for item in keyword_results}
    semantic_by_id = {int(item["product_id"]): item for item in semantic_results}
    fused_by_id = {int(item["id"]): item for item in fused}
    results = []
    for product_id in ids:
        if product_id not in products:
            continue
        row = products[product_id]
        keyword = keyword_by_id.get(product_id)
        semantic = semantic_by_id.get(product_id)
        matched_words = keyword["matched_terms"] if keyword else []
        similarity_note = f"semantic similarity {float(semantic['semantic_score']):.3f}" if semantic else "no semantic match"
        why = f"matched words: {', '.join(matched_words) if matched_words else 'none'}; {similarity_note}"
        fused_item = fused_by_id[product_id]
        results.append(
            {
                "id": row[0],
                "title": row[1],
                "brand": row[2],
                "category": row[3],
                "price": row[4],
                "rating": row[5],
                "in_stock": row[6],
                "keyword_rank": fused_item["keyword_rank"],
                "semantic_rank": fused_item["semantic_rank"],
                "rrf_score": fused_item["rrf_score"],
                "why": why,
            }
        )
        if len(results) == limit:
            break
    return results
