import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DB_PATH = ROOT / "products.db"
MODELS_DIR = ROOT / "models"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


def ensure_backend_table(conn: sqlite3.Connection) -> None:
    # Store the active backend beside embeddings so the API can report it later.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_backend (backend TEXT NOT NULL, model TEXT NOT NULL)"
    )


def catalogue_rows(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    # Embed the same title-plus-description text for every product on each rebuild.
    return conn.execute("SELECT id, title || ' ' || description FROM products ORDER BY id").fetchall()


def fastembed_vectors(texts: list[str]) -> np.ndarray:
    # Use the verified fastembed 0.9.1 TextEmbedding API and cache files under ./models.
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODELS_DIR))
    return np.asarray(list(model.embed(texts)), dtype=np.float32)


def tfidf_vectors(texts: list[str]) -> tuple[np.ndarray, object]:
    # Keep a local, deterministic fallback when the embedding model cannot load.
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return vectors, vectorizer


def write_embeddings(conn: sqlite3.Connection, rows: list[tuple[int, str]], vectors: np.ndarray, backend: str) -> None:
    # Store compact float32 blobs and replace old rows so rebuilds cannot leave stale vectors.
    conn.execute("DELETE FROM product_embeddings")
    conn.execute("DELETE FROM embedding_backend")
    conn.executemany(
        "INSERT INTO product_embeddings (product_id, embedding, model) VALUES (?, ?, ?)",
        [(product_id, vector.astype(np.float32).tobytes(), MODEL_NAME if backend == "fastembed" else "tfidf") for (product_id, _), vector in zip(rows, vectors)],
    )
    conn.execute("INSERT INTO embedding_backend (backend, model) VALUES (?, ?)", (backend, MODEL_NAME if backend == "fastembed" else "tfidf"))


def build_embeddings(db_path: Path = DB_PATH) -> str:
    # Prefer fastembed, but complete the build with TF-IDF if model loading fails.
    MODELS_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        ensure_backend_table(conn)
        rows = catalogue_rows(conn)
        texts = [text for _, text in rows]
        try:
            vectors = fastembed_vectors(texts)
            backend = "fastembed"
        except Exception as error:
            print(f"fastembed unavailable, using TF-IDF fallback: {error}")
            vectors, vectorizer = tfidf_vectors(texts)
            import joblib

            joblib.dump(vectorizer, TFIDF_MODEL_PATH)
            backend = "tfidf"
        write_embeddings(conn, rows, vectors, backend)
        print(f"embedding backend: {backend}; vectors: {len(rows)}; dimensions: {vectors.shape[1]}")
        return backend


if __name__ == "__main__":
    build_embeddings()
