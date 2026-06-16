from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
from openai import AsyncOpenAI
import os

from app.vector_search import VectorSearch
from app.rag_pipeline import RAGPipeline
from app.models import Verse, SearchResult

limiter = Limiter(key_func=get_remote_address)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="en", pattern="^(kn|en)$")
    top_k: int = Field(default=3, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Verse]
    cross_references: list[dict]
    suggested_questions: list[str] = []


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    themes: Optional[list[str]] = None
    top_k: int = Field(default=10, ge=1, le=50)


vector_search: VectorSearch
rag_pipeline: RAGPipeline
openai_client: AsyncOpenAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_search, rag_pipeline, openai_client
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai_client = AsyncOpenAI(api_key=openai_key)
    vector_search = VectorSearch(qdrant_url=qdrant_url)
    rag_pipeline = RAGPipeline(vector_search=vector_search, openai_key=openai_key)
    yield


app = FastAPI(title="Kagga RAG API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def check_moderation(text: str) -> None:
    response = await openai_client.moderations.create(input=text)
    result = response.results[0]
    if result.flagged:
        raise HTTPException(
            status_code=400,
            detail="Your question was flagged by our content filter. Please rephrase and try again."
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit("5/minute")
async def ask(request: Request, body: AskRequest):
    await check_moderation(body.question)
    try:
        result = await rag_pipeline.ask(
            question=body.question,
            language=body.language,
            top_k=body.top_k,
        )
        return AskResponse(**result)
    except HTTPException:
        raise
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