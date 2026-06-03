"""FastAPI entrypoint for the OpenAI Brand Monitor dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.data_loader import load_all
from backend.endpoints import entities, evidence, influencers, overview, posts, qa, topics


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield


app = FastAPI(title="OpenAI Brand Monitor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview.router, prefix="/api", tags=["overview"])
app.include_router(topics.router, prefix="/api", tags=["topics"])
app.include_router(entities.router, prefix="/api", tags=["entities"])
app.include_router(influencers.router, prefix="/api", tags=["influencers"])
app.include_router(posts.router, prefix="/api", tags=["posts"])
app.include_router(evidence.router, prefix="/api", tags=["evidence"])
app.include_router(qa.router, prefix="/api", tags=["qa"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
