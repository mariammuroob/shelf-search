import asyncio
import json
import sqlite3
from pathlib import Path

import httpx

import api


ROOT = Path(__file__).parents[1]


async def request_searches():
    # Explicit lifespan and a bounded ASGI transport avoid the local synchronous client hang.
    async with api.lifespan(api.app):
        encoder = api.app.state.search["encoder"]
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=20) as client:
            with sqlite3.connect(ROOT / "products.db") as conn:
                before = conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
                products_before = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            first = await client.get("/api/search", params={"q": "sturdy organisers", "user_id": "synthetic-a"})
            second = await client.get("/api/search", params={"q": "sturdy organisers", "user_id": "synthetic-b"})
            injection = await client.get("/api/search", params={"q": "x' OR 1=1 --"})
            nonsense = await client.get("/api/search", params={"q": "zzzxqv blorf"})
            long_query = await client.get("/api/search", params={"q": "x" * 201})
            eval_response = await client.get("/api/eval")
            health = await client.get("/api/health")
            root = await client.get("/")
            logs = await client.get("/api/log")
            assert encoder is api.app.state.search["encoder"]
            with sqlite3.connect(ROOT / "products.db") as conn:
                after = conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
                products_after = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert first.status_code == second.status_code == injection.status_code == nonsense.status_code == 200
    assert first.json()["weight"] == 0.25
    assert [(r["id"], r["price"]) for r in first.json()["hybrid_results"]] == [(r["id"], r["price"]) for r in second.json()["hybrid_results"]]
    assert products_before == products_after == 600
    assert all(item["type"] != "outside_filters" or item["product"]["id"] > 0 for item in nonsense.json()["suggestions"])
    assert nonsense.json()["suggestions"]
    assert after == before + 4
    assert long_query.status_code == 422
    assert eval_response.content == (ROOT / "results.json").read_bytes()
    assert health.json()["backend"] in ("fastembed", "tfidf")
    assert root.status_code == 200 and "Synthetic catalogue" in root.text
    assert logs.status_code == 200 and logs.json()


def test_api_guardrails():
    asyncio.run(asyncio.wait_for(request_searches(), timeout=90))
