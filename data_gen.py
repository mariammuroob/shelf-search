import json
import random
import re
import sqlite3
from pathlib import Path


SEED = 42
PRODUCT_COUNT_PER_CATEGORY = 75
DB_PATH = Path(__file__).parent / "products.db"
QUERIES_PATH = Path(__file__).parent / "test_queries.json"

CATEGORIES = ["home", "apparel", "toys", "kitchen", "electronics", "beauty", "grocery", "sports"]

# Keep tags controlled so relevance judgements are reproducible and explainable.
CATEGORY_BUNDLES = {
    "home": [("cosy", "soft", "winter"), ("sturdy", "durable", "organised"), ("kidsroom", "cute", "gift"), ("travel", "compact", "gift"), ("relax", "cosy", "soft")],
    "apparel": [("soft", "winter", "cosy"), ("durable", "travel", "sturdy"), ("gym", "breathable", "active"), ("cute", "gift", "soft"), ("outdoor", "warm", "durable")],
    "toys": [("cute", "kidsroom", "gift"), ("sturdy", "durable", "kidsroom"), ("travel", "compact", "kidsroom"), ("soft", "cosy", "gift"), ("active", "outdoor", "sturdy")],
    "kitchen": [("sturdy", "durable", "organised"), ("gift", "cute", "compact"), ("travel", "compact", "durable"), ("cosy", "warm", "gift"), ("active", "healthy", "sturdy")],
    "electronics": [("travel", "compact", "durable"), ("gym", "active", "sturdy"), ("gift", "cute", "compact"), ("organised", "sturdy", "durable"), ("outdoor", "travel", "warm")],
    "beauty": [("soft", "cosy", "gift"), ("travel", "compact", "organised"), ("cute", "gift", "warm"), ("durable", "sturdy", "active"), ("relax", "soft", "healthy")],
    "grocery": [("healthy", "warm", "cosy"), ("gift", "cute", "healthy"), ("travel", "compact", "healthy"), ("active", "gym", "healthy"), ("durable", "sturdy", "organised")],
    "sports": [("gym", "active", "durable"), ("outdoor", "sturdy", "durable"), ("travel", "compact", "active"), ("winter", "warm", "outdoor"), ("gift", "cute", "active")],
}

CATALOGUE_WORDS = {
    "home": {"items": ["throw", "cushion", "storage basket", "lamp", "duvet cover"], "adjectives": ["faux fur", "woven", "soft-touch", "warm", "neat"]},
    "apparel": {"items": ["hoodie", "jacket", "scarf", "joggers", "socks"], "adjectives": ["fleece", "layering", "lightweight", "everyday", "thermal"]},
    "toys": {"items": ["building set", "plush animal", "puzzle", "craft kit", "remote car"], "adjectives": ["colourful", "playful", "wooden", "creative", "classic"]},
    "kitchen": {"items": ["mixing bowl", "lunch box", "pan", "mug", "spice rack"], "adjectives": ["ceramic", "non-stick", "stackable", "insulated", "handy"]},
    "electronics": {"items": ["desk speaker", "charging stand", "headphones", "power bank", "reading light"], "adjectives": ["wireless", "portable", "quiet", "bright", "fast"]},
    "beauty": {"items": ["face cleanser", "body lotion", "hair brush", "bath set", "lip balm"], "adjectives": ["gentle", "refreshing", "daily", "silky", "calming"]},
    "grocery": {"items": ["tea blend", "granola", "coffee beans", "snack box", "olive oil"], "adjectives": ["roasted", "wholesome", "crunchy", "rich", "small-batch"]},
    "sports": {"items": ["yoga mat", "training bottle", "gym towel", "running belt", "camp chair"], "adjectives": ["grippy", "lightweight", "quick-dry", "padded", "weather-ready"]},
}

BRANDS = ["Northstar", "Pine & Loom", "Brightside", "Cedar Lane", "Bluebird", "Mosaic House"]
MATERIALS = ["brushed cotton", "matte ceramic", "recycled fibre", "powder-coated steel", "clear acrylic", "soft-touch knit"]
USE_CASES = ["weekday routines", "small spaces", "weekend plans", "quiet evenings", "busy households", "everyday errands"]
TITLE_COLOURS = ["amber", "slate", "moss", "coral", "ivory", "navy"]
TITLE_SIZES = ["slim", "standard", "roomy", "mini", "wide"]
TITLE_MATERIALS = ["cotton", "ceramic", "steel", "bamboo", "glass", "canvas"]
TITLE_PACKS = ["single", "2-pack", "3-piece", "4-count"]


def create_schema(connection: sqlite3.Connection) -> None:
    # Use one explicit schema so every phase shares the same durable database contract.
    connection.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            brand TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            in_stock INTEGER NOT NULL,
            rating REAL NOT NULL
        );
        CREATE TABLE product_tags (
            product_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        CREATE VIRTUAL TABLE product_fts USING fts5(
            title, description, brand, category,
            content='products', content_rowid='id'
        );
        CREATE TABLE product_embeddings (
            product_id INTEGER PRIMARY KEY,
            embedding BLOB,
            model TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        CREATE TABLE relevance_judgements (
            query_id TEXT NOT NULL,
            product_id INTEGER NOT NULL,
            relevant INTEGER NOT NULL,
            PRIMARY KEY (query_id, product_id)
        );
        CREATE TABLE search_log (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            filters TEXT NOT NULL,
            top5_ids TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            fallback_used INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def product_copy(category: str, index: int, tags: tuple[str, ...], rng: random.Random) -> tuple[str, str]:
    # Synonym pools make the synthetic catalogue useful for both lexical and intent-oriented tests.
    words = CATALOGUE_WORDS[category]
    adjective = rng.choice(words["adjectives"])
    item = words["items"][index % len(words["items"])]
    colour = TITLE_COLOURS[(index + len(category)) % len(TITLE_COLOURS)]
    size = TITLE_SIZES[(index * 2 + len(category)) % len(TITLE_SIZES)]
    title_material = TITLE_MATERIALS[(index * 3 + len(category)) % len(TITLE_MATERIALS)]
    pack = TITLE_PACKS[(index + len(category)) % len(TITLE_PACKS)]
    title = f"{colour.title()} {size.title()} {title_material.title()} {adjective.title()} {item.title()} ({pack})"
    tag_phrase = ", ".join(tags[:2])
    material = MATERIALS[(index + len(category)) % len(MATERIALS)]
    use_case = USE_CASES[(index * 2 + len(category)) % len(USE_CASES)]
    variant = f"Style {index + 1:02d}"
    sentence_style = index % 3
    if sentence_style == 0:
        description = f"{variant} uses {material} for a {adjective} finish. This {item} suits {use_case}, with a practical shape for {tag_phrase} needs."
    elif sentence_style == 1:
        description = f"Made for {use_case}, this {adjective} {item} brings a {material} feel to daily use. Choose it when {tag_phrase} details matter. {variant}."
    else:
        description = f"A {material} {item} with a {adjective} finish, designed around {tag_phrase} needs. Keep it close for {use_case}; this is {variant}."
    return title, description


def generate_products(connection: sqlite3.Connection, rng: random.Random) -> None:
    # Cycle through balanced tag bundles so every answerable fixture has enough matches.
    product_id = 1
    for category in CATEGORIES:
        for index in range(PRODUCT_COUNT_PER_CATEGORY):
            bundle = CATEGORY_BUNDLES[category][index % len(CATEGORY_BUNDLES[category])]
            tags = tuple(bundle[: 2 + (index % 3 == 0)])
            title, description = product_copy(category, index, tags, rng)
            brand = BRANDS[(product_id + index) % len(BRANDS)]
            price = round(rng.uniform(3, 120), 2)
            in_stock = int(rng.random() < 0.85)
            rating = round(rng.uniform(2.5, 5), 1)
            connection.execute(
                "INSERT INTO products (id, title, description, brand, category, price, in_stock, rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (product_id, title, description, brand, category, price, in_stock, rating),
            )
            connection.executemany(
                "INSERT INTO product_tags (product_id, tag) VALUES (?, ?)",
                [(product_id, tag) for tag in tags],
            )
            product_id += 1
    connection.execute("INSERT INTO product_fts (product_fts) VALUES ('rebuild')")


def build_queries() -> list[dict[str, object]]:
    # Keep fixtures human-readable; tags, not guessed text similarity, define relevance.
    queries = [
        ("q01", "soften the sofa with a snug layer", ["cosy", "soft"], "home", "normal"),
        ("q02", "sturdy organisers under $30", ["sturdy", "organised"], "home", "normal"),
        ("q03", "make a youngster's corner cheerful", ["cute", "kidsroom"], "toys", "normal"),
        ("q04", "compact travel clothes", ["durable", "travel"], "apparel", "normal"),
        ("q05", "warm winter layers", ["soft", "winter"], "apparel", "normal"),
        ("q06", "breathable gym clothing", ["gym", "active"], "apparel", "normal"),
        ("q07", "soft plush gift ideas", ["soft", "gift"], "toys", "normal"),
        ("q08", "durable toys for kids", ["durable", "kidsroom"], "toys", "normal"),
        ("q09", "keep a little one occupied on a trip", ["travel", "compact"], "toys", "normal"),
        ("q10", "stackable kitchen storage", ["compact", "durable"], "kitchen", "normal"),
        ("q11", "cute ceramic gifts", ["cute", "gift"], "kitchen", "normal"),
        ("q12", "healthy cooking helpers", ["healthy", "sturdy"], "kitchen", "normal"),
        ("q13", "portable charging gear", ["compact", "travel"], "electronics", "normal"),
        ("q14", "durable desk setup", ["durable", "organised"], "electronics", "normal"),
        ("q15", "small wireless gifts", ["gift", "compact"], "electronics", "normal"),
        ("q16", "a pampering ritual away from home", ["compact", "travel"], "beauty", "normal"),
        ("q17", "calming self care gifts", ["relax", "soft"], "beauty", "normal"),
        ("q18", "daily beauty under $20", ["compact", "travel"], "beauty", "normal"),
        ("q19", "something nourishing to start the day", ["healthy", "warm"], "grocery", "normal"),
        ("q20", "small travel food box", ["travel", "compact"], "grocery", "normal"),
        ("q21", "wholesome gifts", ["healthy", "gift"], "grocery", "normal"),
        ("q22", "grippy gym equipment", ["gym", "active"], "sports", "normal"),
        ("q23", "weather ready winter gear", ["winter", "outdoor"], "sports", "normal"),
        ("q24", "lightweight active travel kit", ["compact", "travel"], "sports", "normal"),
        ("q25", "cosy home thows", ["cosy", "soft"], "home", "awkward"),
        ("q26", "something sturdy", ["sturdy", "durable"], "kitchen", "awkward"),
        ("q27", "GIFT IDEAS", ["gift", "cute"], "toys", "awkward"),
        ("q28", "warm", ["soft", "winter"], "apparel", "awkward"),
        ("q29", "printer ink cartridges", ["printer"], "electronics", "off_catalogue"),
        ("q30", "camping tents", ["camping"], "sports", "off_catalogue"),
    ]
    return [
        {"query_id": query_id, "query": query, "required_tags": tags, "category": category, "kind": kind}
        for query_id, query, tags, category, kind in queries
    ]


def fill_judgements(connection: sqlite3.Connection, queries: list[dict[str, object]]) -> dict[str, int]:
    # Evaluate relevance from the stored hidden tags and category constraint only.
    counts: dict[str, int] = {}
    for query in queries:
        required_tags = set(query["required_tags"])
        rows = connection.execute("SELECT products.id, products.category FROM products").fetchall()
        relevant_ids = []
        for product_id, category in rows:
            product_tags = {
                tag for (tag,) in connection.execute(
                    "SELECT tag FROM product_tags WHERE product_id = ?", (product_id,)
                ).fetchall()
            }
            relevant = int(category == query["category"] and required_tags.issubset(product_tags))
            connection.execute(
                "INSERT INTO relevance_judgements (query_id, product_id, relevant) VALUES (?, ?, ?)",
                (query["query_id"], product_id, relevant),
            )
            if relevant:
                relevant_ids.append(product_id)
        counts[str(query["query_id"])] = len(relevant_ids)
    return counts


def token_set(text: str) -> set[str]:
    # Ignore stopwords and numeric price tokens when measuring lexical separation.
    stopwords = {"a", "an", "and", "for", "home", "ideas", "the", "to", "under", "with"}
    return {token for token in re.findall(r"[a-z]+", text.lower()) if token not in stopwords}


def no_overlap_share(connection: sqlite3.Connection, queries: list[dict[str, object]]) -> float:
    answerable = [query for query in queries if query["kind"] != "off_catalogue"]
    no_overlap = 0
    for query in answerable:
        query_tokens = token_set(str(query["query"]))
        rows = connection.execute(
            "SELECT products.title, products.description, products.brand, products.category FROM products JOIN relevance_judgements ON products.id = relevance_judgements.product_id WHERE relevance_judgements.query_id = ? AND relevance_judgements.relevant = 1",
            (query["query_id"],),
        ).fetchall()
        product_tokens = set().union(*(token_set(" ".join(row)) for row in rows))
        if not query_tokens.intersection(product_tokens):
            no_overlap += 1
    return no_overlap / len(answerable)


def print_summary(connection: sqlite3.Connection, queries: list[dict[str, object]], counts: dict[str, int]) -> None:
    # Print compact evidence that generation and judgement invariants held.
    row_counts = {
        "products": connection.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        "product_tags": connection.execute("SELECT COUNT(*) FROM product_tags").fetchone()[0],
        "relevance_judgements": connection.execute("SELECT COUNT(*) FROM relevance_judgements").fetchone()[0],
    }
    price_range = connection.execute("SELECT MIN(price), MAX(price) FROM products").fetchone()
    print(f"row counts: {row_counts}")
    print(f"price range: ${price_range[0]:.2f} to ${price_range[1]:.2f}")
    print(f"query relevance counts: {counts}")
    print(f"answerable queries with at least 5 relevant products: {sum(count >= 5 for query_id, count in counts.items() if next(q for q in queries if q['query_id'] == query_id)['kind'] != 'off_catalogue')}/28")
    print(f"no-word-overlap share among answerable queries: {no_overlap_share(connection, queries):.1%}")


def main() -> None:
    # Delete only this generated artifact so reruns are clean and reproducible.
    if DB_PATH.exists():
        DB_PATH.unlink()
    rng = random.Random(SEED)
    queries = build_queries()
    QUERIES_PATH.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    with sqlite3.connect(DB_PATH) as connection:
        create_schema(connection)
        generate_products(connection, rng)
        counts = fill_judgements(connection, queries)
        answerable_counts = [counts[str(query["query_id"])] for query in queries if query["kind"] != "off_catalogue"]
        if min(answerable_counts) < 5:
            raise RuntimeError("an answerable query has fewer than five relevant products")
        print_summary(connection, queries, counts)


if __name__ == "__main__":
    main()
