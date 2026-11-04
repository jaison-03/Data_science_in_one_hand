"""Typed, read-only API for the React project explorer."""

from typing import Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.catalog import PROJECT_ROOT, build_catalog, filter_catalog

app = FastAPI(
    title="AI Project Explorer API",
    version="1.0.0",
    description="Read-only catalog of the existing AI, machine-learning, and NLP projects.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["GET"],
    allow_headers=["Accept"],
)

_catalog = build_catalog()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/catalog")
def get_catalog(
    q: str = Query(default="", max_length=200),
    area: str | None = Query(default=None, max_length=80),
    sort: Literal["title", "area", "category"] = "title",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=12, ge=1, le=100),
) -> dict[str, object]:
    if area is not None and area not in {item["area"] for item in _catalog}:
        raise HTTPException(status_code=422, detail="Unknown project area.")

    filtered = filter_catalog(_catalog, query=q, area=area, sort=sort)
    return {
        "items": filtered[offset : offset + limit],
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "areas": sorted({item["area"] for item in _catalog}),
    }


@app.get("/api/catalog/{project_id}")
def get_project(project_id: str) -> dict[str, str | None]:
    project = next((item for item in _catalog if item["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


_dist = PROJECT_ROOT / "dist"
if _dist.is_dir():
    _dist_root = _dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        candidate = (_dist / Path(path)).resolve()
        if candidate != _dist_root and _dist_root not in candidate.parents:
            raise HTTPException(status_code=404, detail="Asset not found.")
        if not candidate.is_file():
            candidate = _dist_root / "index.html"
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Frontend build not found.")
        return FileResponse(candidate)
