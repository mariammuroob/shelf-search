from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


# Keep the first phase intentionally small: health is the only API contract yet.
app = FastAPI()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve the static placeholder from the repository's documented frontend folder.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
