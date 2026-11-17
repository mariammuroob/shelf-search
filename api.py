import difflib
import json
import math
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from search import keyword_search, parse_query, search_terms


ROOT = Path(__file__).parent
DB = ROOT / "products.db"
CATEGORIES = ("home", "apparel", "toys", "kitchen", "electronics", "beauty", "grocery", "sports")


def load_state():
    # Load the encoder and normalized catalogue matrix once per worker at startup.
    with sqlite3.connect(DB) as conn:
        backend, model = conn.execute("SELECT backend, model FROM embedding_backend LIMIT 1").fetchone()
        rows = conn.execute("SELECT product_id, embedding FROM product_embeddings ORDER BY product_id").fetchall()
        vocabulary = sorted({word for (description,) in conn.execute("SELECT title || ' ' || description FROM products") for word in search_terms(description)})
    ids = np.asarray([row[0] for row in rows])
    matrix = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    if backend == "fastembed":
        from fastembed import TextEmbedding
        encoder = TextEmbedding(model_name=model, cache_dir=str(ROOT / "models"))
    else:
        import joblib
        encoder = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    weight = float(json.loads((ROOT / "dev_selection.json").read_text(encoding="utf-8"))["selected_weight"])
    return {"backend": backend, "encoder": encoder, "ids": ids, "matrix": matrix, "vocabulary": vocabulary, "weight": weight}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search = load_state()
    yield
    del app.state.search


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def safe_error(_request: Request, error: Exception):
    # Never expose a traceback to callers.
    return JSONResponse({"error": error.detail if isinstance(error, HTTPException) else "Search is temporarily unavailable"}, status_code=error.status_code if isinstance(error, HTTPException) else 500)


def semantic_candidates(state, text):
    # Match frozen cosine ranking using the startup matrix and one query encoding.
    encoder = state["encoder"]
    vector = np.asarray(next(encoder.query_embed(text)) if state["backend"] == "fastembed" else encoder.transform([text]).toarray()[0], dtype=np.float32)
    norm = np.linalg.norm(vector)
    scores = state["matrix"] @ (vector / norm) if norm else np.zeros(len(state["ids"]))
    return sorted(zip(state["ids"].tolist(), scores.tolist()), key=lambda pair: (-pair[1], pair[0]))[:200]


def allowed_products(conn, ids, price, category, in_stock):
    # Bind every filter and each candidate id through generated IN placeholders.
    if not ids:
        return {}
    clauses = ["id IN (" + ", ".join("?" for _ in ids) + ")"]
    params = list(ids)
    if price is not None:
        clauses.append("price <= ?")
        params.append(price)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if in_stock:
        clauses.append("in_stock = ?")
        params.append(1)
    rows = conn.execute("SELECT id, title, brand, category, price, rating, in_stock FROM products WHERE " + " AND ".join(clauses), params)
    return {row[0]: row for row in rows}


def present(row, keyword_rank, semantic_rank, score, words, cosine):
    return {"id": row[0], "title": row[1], "brand": row[2], "category": row[3], "price": row[4], "rating": row[5], "in_stock": bool(row[6]), "keyword_rank": keyword_rank, "semantic_rank": semantic_rank, "rrf_score": score, "why": f"matched words: {', '.join(words) if words else 'none'}; " + (f"semantic similarity {cosine:.3f}" if cosine is not None else "no semantic match")}


def ranked_results(conn, state, text, price, category, in_stock):
    # Keep the 200 candidate arms and dev-selected RRF weight fixed.
    keyword = keyword_search(conn, text, None, None, False, k=200)
    semantic = semantic_candidates(state, text)
    kr = {item["product_id"]: rank for rank, item in enumerate(keyword, 1)}
    sr = {pid: rank for rank, (pid, _) in enumerate(semantic, 1)}
    cosines = dict(semantic)
    scores = {pid: 1 / (60 + rank) for pid, rank in kr.items()}
    for pid, rank in sr.items():
        scores[pid] = scores.get(pid, 0) + state["weight"] / (60 + rank)
    products = allowed_products(conn, list(scores), price, category, in_stock)
    ordered = sorted(products, key=lambda pid: (-scores[pid], pid))[:5]
    hybrid = [present(products[pid], kr.get(pid), sr.get(pid), scores[pid], search_terms(text) if pid in kr else [], cosines.get(pid)) for pid in ordered]
    filtered = keyword_search(conn, text, price, category, in_stock, k=5)
    keyword_rows = allowed_products(conn, [item["product_id"] for item in filtered], price, category, in_stock)
    baseline = [present(keyword_rows[item["product_id"]], kr[item["product_id"]], None, None, item["matched_terms"], None) for item in filtered]
    return baseline, hybrid, semantic


def suggestions(conn, state, text, price, category, in_stock, semantic):
    # Relax filters only to suggest real products and label any outside-filter examples.
    items = []
    if price is not None and any(ranked_results(conn, state, text, price * 1.5, category, in_stock)[:2]):
        items.append({"type": "relax_price", "max_price": price * 1.5})
    if category and any(ranked_results(conn, state, text, price, None, in_stock)[:2]):
        items.append({"type": "remove_category"})
    words = []
    for term in search_terms(text):
        words.extend(difflib.get_close_matches(term, state["vocabulary"], n=1, cutoff=0.7))
    items.extend({"type": "spelling", "text": word} for word in dict.fromkeys(words) if word not in search_terms(text))
    in_stock_rows = allowed_products(conn, [pid for pid, _ in semantic], None, None, True)
    for pid, cosine in semantic:
        if pid in in_stock_rows:
            items.append({"type": "outside_filters", "label": "outside your filters", "product": present(in_stock_rows[pid], None, None, None, [], cosine)})
        if sum(item["type"] == "outside_filters" for item in items) == 3:
            break
    return items


def log_search(query, filters, ids, latency, fallback):
    # A missing, read-only, or recreated log database must not fail the search.
    try:
        with sqlite3.connect(DB, timeout=0.2) as conn:
            conn.execute("INSERT INTO search_log (query, filters, top5_ids, latency_ms, fallback_used, created_at) VALUES (?, ?, ?, ?, ?, ?)", (query, json.dumps(filters), json.dumps(ids), latency, int(fallback), datetime.now(timezone.utc).isoformat()))
    except sqlite3.Error:
        pass


@app.get("/api/search")
def api_search(request: Request, q: str, max_price: float | None = None, category: str | None = None, in_stock: bool = False):
    started = time.perf_counter()
    if not q.strip() or len(q) > 200:
        raise HTTPException(422, "q must contain 1 to 200 characters")
    if category is not None and category not in CATEGORIES:
        raise HTTPException(422, "unknown category")
    if max_price is not None and (not math.isfinite(max_price) or max_price <= 0):
        raise HTTPException(422, "max_price must be positive")
    text, parsed_price = parse_query(q)
    price = max_price if max_price is not None else parsed_price
    if price is not None and (not math.isfinite(price) or price <= 0):
        raise HTTPException(422, "max_price must be positive")
    if not text.strip():
        raise HTTPException(422, "q must include search terms")
    state = request.app.state.search
    with sqlite3.connect(DB) as conn:
        keyword, hybrid, semantic = ranked_results(conn, state, text, price, category, in_stock)
        suggestions_list = suggestions(conn, state, text, price, category, in_stock, semantic) if not keyword or not hybrid else []
    filters = {"max_price": price, "category": category, "in_stock": in_stock}
    latency = (time.perf_counter() - started) * 1000
    log_search(q, filters, [item["id"] for item in hybrid], latency, state["backend"] == "tfidf")
    return {"keyword_results": keyword, "hybrid_results": hybrid, "parsed_filters": filters, "no_results": not hybrid, "suggestions": suggestions_list, "latency_ms": latency, "backend": state["backend"], "weight": state["weight"]}


@app.get("/api/eval")
def api_eval():
    # Return the committed bytes without evaluating again.
    return Response((ROOT / "results.json").read_bytes(), media_type="application/json")


@app.get("/api/log")
def api_log():
    try:
        with sqlite3.connect(DB) as conn:
            rows = conn.execute("SELECT id, query, filters, top5_ids, latency_ms, fallback_used, created_at FROM search_log ORDER BY id DESC LIMIT ?", (20,)).fetchall()
    except sqlite3.Error:
        rows = []
    return [{"id": r[0], "query": r[1], "filters": json.loads(r[2]), "top5_ids": json.loads(r[3]), "latency_ms": r[4], "fallback_used": bool(r[5]), "created_at": r[6]} for r in rows]


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "backend": request.app.state.search["backend"]}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
