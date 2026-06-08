from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os

from app.vector_search import VectorSearch
from app.rag_pipeline import RAGPipeline
from app.models import Verse, SearchResult


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="en", pattern="^(kn|en)$")
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Verse]
    cross_references: list[dict]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    themes: Optional[list[str]] = None
    top_k: int = Field(default=10, ge=1, le=50)


vector_search: VectorSearch
rag_pipeline: RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_search, rag_pipeline
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    vector_search = VectorSearch(qdrant_url=qdrant_url)
    rag_pipeline = RAGPipeline(vector_search=vector_search, openai_key=openai_key)
    yield


app = FastAPI(title="Kagga RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        result = await rag_pipeline.ask(
            question=request.question,
            language=request.language,
            top_k=request.top_k,
        )
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    try:
        results = await vector_search.search(
            query=request.query,
            themes=request.themes,
            top_k=request.top_k,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))