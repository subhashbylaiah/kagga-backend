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
            detail="We weren't able to process that question. Please try again."
        )


async def check_topic(text: str) -> None:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Is this a question where a person is seeking wisdom, meaning, coping strategies, "
                    "emotional guidance, or philosophical reflection — even if framed in worldly terms "
                    "like work, money, relationships, or daily life? "
                    "Answer YES if there is any human experience or emotional angle. "
                    "Answer NO only if it is purely technical, factual trivia, a coding task, or has "
                    "no connection to human feelings or life experience whatsoever.\n\n"
                    f"Question: {text}"
                ),
            }
        ],
        temperature=0,
        max_tokens=5,
    )
    answer = (response.choices[0].message.content or "").strip().upper()
    if answer != "YES":
        raise HTTPException(
            status_code=400,
            detail="Kagga speaks to the human experience — grief, purpose, relationships, mortality, and meaning. Please ask something along those lines."
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/suggestions")
async def suggestions():
    return [
        "How do I deal with uncertainty?",
        "What does Kagga say about grief?",
        "How to find meaning in suffering?",
        "What is the purpose of life according to Kagga?",
        "How does Kagga view the nature of God?",
    ]


@app.get("/facts")
async def facts():
    return [
        "DVG never completed formal higher education — he taught himself Sanskrit and English, and went on to write one of Kannada literature's greatest works.",
        "Mankutimmana Kagga was published in 1943, when DVG was 56 years old — the result of decades of quiet observation.",
        "Mankutimma is not a real person — DVG was talking to himself, questioning his own understanding of life.",
        "The word 'Kagga' loosely means 'ramblings' — DVG deliberately chose a humble title for a profound work.",
        "Each verse ends with 'Mankutimma' — meaning 'you fool' — DVG's gentle reminder to himself to stay humble.",
        "Mankutimmana Kagga has been translated into English, Hindi, and Sanskrit.",
        "DVG was a journalist before he was a poet — he ran Kannada and English newspapers in the early 1900s.",
        "DVG lived through two World Wars and India's independence — all of which shaped Kagga's themes of uncertainty and impermanence.",
        "DVG received the Padma Bhushan from the Government of India in 1974, just one year before his death at age 88.",
        "Mankutimmana Kagga is still taught in Karnataka schools and colleges, decades after it was written.",
        "DVG wrote a memoir called Jnapaka Chitrashale — vivid portraits of people and life in old Mysore, considered a masterpiece of Kannada prose.",
        "DVG founded the Gokhale Institute of Public Affairs in Bengaluru in 1945 as a free, open space for civic dialogue — anyone, regardless of status, was welcome.",
        "When Karnataka honored DVG with ₹90,000 in 1970, he donated the entire sum to the Gokhale Institute rather than keep a single rupee for himself.",
        "DVG profiled both the famous and the ordinary in his biographical works — he believed every life had equal dignity and deserved to be remembered.",
        "DVG translated Shakespeare's Macbeth into Kannada — he believed great literature had no borders.",
        "DVG also wrote children's literature, wanting wisdom to reach the youngest readers too.",
        "DVG presided over the 18th Kannada Sahitya Sammelana in 1932 — one of the highest honors in Kannada literary culture.",
        "India Post issued a commemorative stamp in DVG's honor in 1988, thirteen years after his death.",
        "DVG's complete works were compiled into eleven volumes and are still published in e-book formats today by the Gokhale Institute.",
        "DVG wrote the first Kannada-language book on political science in 1952, drawing on both Western scholarship and Indian philosophy.",
        "DVG launched multiple newspapers between 1906 and 1921, including an English magazine supported by Diwan Visvesvaraya.",
        "DVG lived to 88, passing away on October 7, 1975 — having witnessed a century of change while remaining rooted in timeless wisdom.",
    ]


@app.post("/ask", response_model=AskResponse)
@limiter.limit("5/minute")
async def ask(request: Request, body: AskRequest):
    await check_moderation(body.question)
    await check_topic(body.question)
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