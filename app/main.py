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
async def suggestions(lang: str = "en"):
    if lang == "kn":
        return [
            "ಅನಿಶ್ಚಿತತೆಯನ್ನು ಹೇಗೆ ಎದುರಿಸುವುದು?",
            "ದುಃಖದ ಬಗ್ಗೆ ಕಗ್ಗ ಏನು ಹೇಳುತ್ತದೆ?",
            "ಸಂಕಟದಲ್ಲಿ ಅರ್ಥ ಹೇಗೆ ಕಂಡುಕೊಳ್ಳುವುದು?",
            "ಜೀವನದ ಉದ್ದೇಶವೇನು?",
            "ಕರ್ಮದ ಬಗ್ಗೆ ಕಗ್ಗ ಏನು ಹೇಳುತ್ತದೆ?",
        ]
    return [
        "How do I deal with uncertainty?",
        "What does Kagga say about grief?",
        "How to find meaning in suffering?",
        "What is the purpose of life according to Kagga?",
        "How does Kagga view the nature of God?",
    ]


@app.get("/facts")
async def facts(lang: str = "en"):
    if lang == "kn":
        return [
            "ಡಿ.ವಿ.ಜಿ ಔಪಚಾರಿಕ ಉನ್ನತ ಶಿಕ್ಷಣ ಪಡೆಯಲಿಲ್ಲ — ಸಂಸ್ಕೃತ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಅನ್ನು ತಾವೇ ಕಲಿತರು, ಮತ್ತು ಕನ್ನಡ ಸಾಹಿತ್ಯದ ಶ್ರೇಷ್ಠ ಕೃತಿಗಳಲ್ಲೊಂದನ್ನು ರಚಿಸಿದರು.",
            "ಮಂಕುತಿಮ್ಮನ ಕಗ್ಗ ೧೯೪೩ರಲ್ಲಿ ಪ್ರಕಟವಾಯಿತು, ಆಗ ಡಿ.ವಿ.ಜಿ ೫೬ ವರ್ಷದವರಾಗಿದ್ದರು — ದಶಕಗಳ ಮೌನ ಚಿಂತನೆಯ ಫಲ.",
            "ಮಂಕುತಿಮ್ಮ ಒಬ್ಬ ನಿಜವಾದ ವ್ಯಕ್ತಿಯಲ್ಲ — ಡಿ.ವಿ.ಜಿ ತಮ್ಮಲ್ಲೇ ತಾವು ಮಾತನಾಡಿಕೊಳ್ಳುತ್ತಿದ್ದರು, ತಮ್ಮದೇ ಜೀವನ ತಿಳಿವಳಿಕೆಯನ್ನು ಪ್ರಶ್ನಿಸಿಕೊಳ್ಳುತ್ತಿದ್ದರು.",
            "'ಕಗ್ಗ' ಎಂದರೆ ಸರಳವಾಗಿ 'ಗೊಣಗಾಟ' ಎಂದರ್ಥ — ಡಿ.ವಿ.ಜಿ ಒಂದು ಗಹನ ಕೃತಿಗೆ ಉದ್ದೇಶಪೂರ್ವಕವಾಗಿ ವಿನಮ್ರ ಶೀರ್ಷಿಕೆ ಆರಿಸಿದರು.",
            "ಪ್ರತಿ ಪದ್ಯವೂ 'ಮಂಕುತಿಮ್ಮ' ಎಂದು ಕೊನೆಗೊಳ್ಳುತ್ತದೆ — ಅರ್ಥ 'ಮೂರ್ಖ' ಎಂದು — ಡಿ.ವಿ.ಜಿ ತಮಗೆ ತಾವೇ ನೀಡಿಕೊಂಡ ವಿನಮ್ರತೆಯ ಜ್ಞಾಪನೆ.",
            "ಮಂಕುತಿಮ್ಮನ ಕಗ್ಗ ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ ಮತ್ತು ಸಂಸ್ಕೃತಕ್ಕೆ ಅನುವಾದಗೊಂಡಿದೆ.",
            "ಡಿ.ವಿ.ಜಿ ಕವಿಯಾಗುವ ಮೊದಲು ಪತ್ರಕರ್ತರಾಗಿದ್ದರು — ೧೯೦೦ರ ದಶಕದ ಆರಂಭದಲ್ಲಿ ಕನ್ನಡ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಪತ್ರಿಕೆಗಳನ್ನು ನಡೆಸಿದರು.",
            "ಡಿ.ವಿ.ಜಿ ಎರಡು ವಿಶ್ವಯುದ್ಧಗಳನ್ನು ಮತ್ತು ಭಾರತದ ಸ್ವಾತಂತ್ರ್ಯವನ್ನು ಕಂಡರು — ಇವೆಲ್ಲವೂ ಕಗ್ಗದ ಅನಿಶ್ಚಿತತೆ ಮತ್ತು ಅಸ್ಥಿರತೆಯ ಭಾವನೆಗಳನ್ನು ರೂಪಿಸಿದವು.",
            "ಡಿ.ವಿ.ಜಿ ೧೯೭೪ರಲ್ಲಿ ಭಾರತ ಸರ್ಕಾರದಿಂದ ಪದ್ಮಭೂಷಣ ಪಡೆದರು — ತಮ್ಮ ೮೮ನೇ ವಯಸ್ಸಿನಲ್ಲಿ ನಿಧನರಾಗುವ ಒಂದು ವರ್ಷ ಮೊದಲು.",
            "ಮಂಕುತಿಮ್ಮನ ಕಗ್ಗ ಬರೆದ ದಶಕಗಳ ನಂತರವೂ ಕರ್ನಾಟಕದ ಶಾಲೆಗಳು ಮತ್ತು ಕಾಲೇಜುಗಳಲ್ಲಿ ಇಂದಿಗೂ ಕಲಿಸಲಾಗುತ್ತದೆ.",
            "ಡಿ.ವಿ.ಜಿ 'ಜ್ಞಾಪಕ ಚಿತ್ರಶಾಲೆ' ಎಂಬ ಸ್ಮರಣ ಕೃತಿ ಬರೆದರು — ಹಳೆಯ ಮೈಸೂರಿನ ಜನರ ಜೀವಂತ ಚಿತ್ರಣ, ಕನ್ನಡ ಗದ್ಯದ ಮಹಾಕೃತಿ ಎಂದು ಪರಿಗಣಿಸಲಾಗಿದೆ.",
            "ಡಿ.ವಿ.ಜಿ ೧೯೪೫ರಲ್ಲಿ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಗೋಖಲೆ ಸಾರ್ವಜನಿಕ ವಿಷಯಗಳ ಸಂಸ್ಥೆಯನ್ನು ಸ್ಥಾಪಿಸಿದರು — ಯಾರು ಬೇಕಾದರೂ, ಯಾವ ಸ್ತರದವರಾದರೂ ಬರಬಹುದಾದ ಮುಕ್ತ ನಾಗರಿಕ ಸಂವಾದದ ತಾಣ.",
            "೧೯೭೦ರಲ್ಲಿ ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಡಿ.ವಿ.ಜಿ ಅವರಿಗೆ ₹೯೦,೦೦೦ ಗೌರವ ನೀಡಿದಾಗ, ಅವರು ಆ ಇಡೀ ಮೊತ್ತವನ್ನು ಗೋಖಲೆ ಸಂಸ್ಥೆಗೆ ದಾನ ಮಾಡಿದರು — ತಮಗಾಗಿ ಒಂದು ರೂಪಾಯಿಯೂ ಇಟ್ಟುಕೊಳ್ಳಲಿಲ್ಲ.",
            "ಡಿ.ವಿ.ಜಿ ತಮ್ಮ ಜೀವನಚರಿತ್ರೆ ಕೃತಿಗಳಲ್ಲಿ ಪ್ರಸಿದ್ಧರ ಮತ್ತು ಸಾಮಾನ್ಯರ ಕಥೆಗಳನ್ನು ಸಮಾನವಾಗಿ ಬರೆದರು — ಪ್ರತಿ ಜೀವನಕ್ಕೂ ಸಮಾನ ಘನತೆ ಇದೆ ಎಂದು ಅವರು ನಂಬಿದ್ದರು.",
            "ಡಿ.ವಿ.ಜಿ ಶೇಕ್ಸ್‌ಪಿಯರ್ ಅವರ 'ಮ್ಯಾಕ್‌ಬೆತ್' ಅನ್ನು ಕನ್ನಡಕ್ಕೆ ಅನುವಾದಿಸಿದರು — ಶ್ರೇಷ್ಠ ಸಾಹಿತ್ಯಕ್ಕೆ ಗಡಿಗಳಿಲ್ಲ ಎಂದು ಅವರು ನಂಬಿದ್ದರು.",
            "ಡಿ.ವಿ.ಜಿ ಮಕ್ಕಳ ಸಾಹಿತ್ಯವನ್ನೂ ರಚಿಸಿದರು — ಜ್ಞಾನ ಚಿಕ್ಕ ಮಕ್ಕಳನ್ನೂ ತಲುಪಬೇಕೆಂಬ ಆಸೆಯಿಂದ.",
            "ಡಿ.ವಿ.ಜಿ ೧೯೩೨ರಲ್ಲಿ ೧೮ನೇ ಕನ್ನಡ ಸಾಹಿತ್ಯ ಸಮ್ಮೇಳನದ ಅಧ್ಯಕ್ಷತೆ ವಹಿಸಿದರು — ಕನ್ನಡ ಸಾಹಿತ್ಯ ಸಂಸ್ಕೃತಿಯ ಅತ್ಯುಚ್ಚ ಗೌರವಗಳಲ್ಲಿ ಒಂದು.",
            "ಇಂಡಿಯಾ ಪೋಸ್ಟ್ ೧೯೮೮ರಲ್ಲಿ ಡಿ.ವಿ.ಜಿ ಅವರ ಗೌರವಾರ್ಥ ಸ್ಮರಣಾರ್ಥ ಅಂಚೆಚೀಟಿ ಬಿಡುಗಡೆ ಮಾಡಿತು — ಅವರ ನಿಧನದ ಹದಿಮೂರು ವರ್ಷಗಳ ನಂತರ.",
            "ಡಿ.ವಿ.ಜಿ ಅವರ ಸಮಗ್ರ ಕೃತಿಗಳು ಹನ್ನೊಂದು ಸಂಪುಟಗಳಲ್ಲಿ ಸಂಕಲಿಸಲ್ಪಟ್ಟಿದ್ದು, ಗೋಖಲೆ ಸಂಸ್ಥೆ ಇಂದಿಗೂ ಅವುಗಳನ್ನು ಇ-ಬುಕ್ ರೂಪದಲ್ಲಿ ಪ್ರಕಟಿಸುತ್ತಿದೆ.",
            "ಡಿ.ವಿ.ಜಿ ೧೯೫೨ರಲ್ಲಿ ರಾಜಕೀಯ ವಿಜ್ಞಾನದ ಮೇಲಿನ ಮೊದಲ ಕನ್ನಡ ಪುಸ್ತಕ ಬರೆದರು — ಪಶ್ಚಿಮದ ವಿದ್ಯೆ ಮತ್ತು ಭಾರತೀಯ ತತ್ತ್ವಜ್ಞಾನ ಎರಡನ್ನೂ ಮೇಳೈಸಿ.",
            "ಡಿ.ವಿ.ಜಿ ೧೯೦೬ ರಿಂದ ೧೯೨೧ರ ಮಧ್ಯೆ ಹಲವು ಪತ್ರಿಕೆಗಳನ್ನು ಪ್ರಾರಂಭಿಸಿದರು — ದಿವಾನ್ ವಿಶ್ವೇಶ್ವರಯ್ಯ ಅವರ ಬೆಂಬಲದಿಂದ ಒಂದು ಇಂಗ್ಲಿಷ್ ಪತ್ರಿಕೆ ಸೇರಿದಂತೆ.",
            "ಡಿ.ವಿ.ಜಿ ೮೮ ವರ್ಷ ಬದುಕಿದರು, ಅಕ್ಟೋಬರ್ ೭, ೧೯೭೫ರಂದು ನಿಧನರಾದರು — ಒಂದು ಶತಮಾನದ ಬದಲಾವಣೆಗಳನ್ನು ಕಂಡರೂ ಕಾಲಾತೀತ ಜ್ಞಾನದಲ್ಲಿ ಬೇರೂರಿದ್ದರು.",
        ]
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