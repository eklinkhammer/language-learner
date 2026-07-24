import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import speech, tutor, exercises


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    # TODO: Pre-load ML models here for faster first request
    yield
    # Cleanup on shutdown


app = FastAPI(title="Language Learner", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(speech.router, prefix="/api/speech", tags=["speech"])
app.include_router(tutor.router, prefix="/api/tutor", tags=["tutor"])
app.include_router(exercises.router, prefix="/api/exercises", tags=["exercises"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
