import json
import sqlite3
from pathlib import Path

from data_gen import token_set


ROOT = Path(__file__).parent
# Three new tag combinations per category; the first six use subjective paraphrases.
SPECS = [
    ("a peaceful nook after a long shift", "home", ["relax", "soft"]),
    ("durable baskets for a crowded corner", "home", ["organised", "durable"]),
    ("faux fur accent for lounging", "home", ["relax", "cosy"]),
    ("something for chilly school mornings", "apparel", ["winter", "cosy"]),
    ("outdoor layering for commutes", "apparel", ["outdoor", "durable"]),
    ("active breathable joggers", "apparel", ["active", "breathable"]),
    ("a surprise for a young builder", "toys", ["kidsroom", "sturdy"]),
    ("creative play for trips", "toys", ["compact", "kidsroom"]),
    ("soft animal for bedtime", "toys", ["cosy", "gift"]),
    ("cute kitchen present for a new flat", "kitchen", ["gift", "compact"]),
    ("sturdy cooking for weekday routines", "kitchen", ["healthy", "active"]),
    ("warm mug for quiet evenings", "kitchen", ["warm", "cosy"]),
    ("keep devices ready during a getaway", "electronics", ["travel", "durable"]),
    ("organised desk stand", "electronics", ["organised", "sturdy", "durable"]),
    ("active gym audio", "electronics", ["gym", "sturdy"]),
    ("a little treat after a tiring day", "beauty", ["relax", "healthy"]),
    ("gentle bath for cosy evenings", "beauty", ["soft", "cosy"]),
    ("compact beauty organiser", "beauty", ["compact", "organised"]),
    ("rich tea for winter evenings", "grocery", ["warm", "healthy", "cosy"]),
    ("healthy snack for active days", "grocery", ["gym", "active"]),
    ("compact food for travel", "grocery", ["travel", "healthy"]),
    ("durable outdoor training gear", "sports", ["outdoor", "sturdy"]),
    ("warm equipment for winter walks", "sports", ["winter", "warm", "outdoor"]),
    ("compact kit for active trips", "sports", ["compact", "active"]),
]


def main() -> None:
    # Dev labels use the same catalogue tag rule but never enter test judgements.
    test = json.loads((ROOT / "test_queries.json").read_text(encoding="utf-8"))
    test_pairs = {(q["category"], frozenset(q["required_tags"])) for q in test}
    queries = []
    with sqlite3.connect(ROOT / "products.db") as conn:
        for number, (wording, category, tags) in enumerate(SPECS, 1):
            assert (category, frozenset(tags)) not in test_pairs
            rows = conn.execute("SELECT id, title, description, brand, category FROM products WHERE category = ?", (category,)).fetchall()
            relevant = []
            product_words = set()
            for product_id, title, description, brand, product_category in rows:
                assigned = {row[0] for row in conn.execute("SELECT tag FROM product_tags WHERE product_id = ?", (product_id,))}
                if set(tags).issubset(assigned):
                    relevant.append(product_id)
                    product_words |= token_set(" ".join((title, description, brand, product_category)))
            assert len(relevant) >= 5, (number, tags)
            overlap = bool(token_set(wording) & product_words)
            queries.append({"query_id": f"d{number:02d}", "query": wording, "category": category, "required_tags": tags, "relevant_ids": relevant, "group": "shared_word" if overlap else "no_word_overlap"})
    assert len({q["query"].lower() for q in queries}) == 24
    assert sum(q["group"] == "no_word_overlap" for q in queries) == 6, [(q["query_id"], q["group"]) for q in queries]
    (ROOT / "dev_queries.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    print("dev set: 24 queries, 6 paraphrases; relevant counts:", {q["query_id"]: len(q["relevant_ids"]) for q in queries})


if __name__ == "__main__":
    main()
