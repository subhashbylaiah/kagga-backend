# Kagga RAG Backend

FastAPI + Qdrant + OpenAI backend for Mankutimmana Kagga iOS app.

## Quick Start

### 1. Local Development

```bash
cd backend
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

docker compose up -d
```

API runs at `http://localhost:8000`

### 2. Get & Convert Data

```bash
# Clone the source repo
git clone https://github.com/rakeshmbgit/kagga-website.git ../kagga-website

# Convert TypeScript to JSON
python scripts/convert_kagga_ts.py ../kagga-website
# Output: data/kaggas.json (945 verses)
```

### 3. Embed Verses (one-time)

```bash
docker compose run --rm api python scripts/embed_verses.py
```

### 4. Test

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Kagga say about impermanence?", "language": "en"}'
```

## Data Format

`data/kaggas.json` — array of 945 objects:

```json
{
  "id": 1,
  "kannada_text": "ಕನ್ನಡ ಪಠ್ಯ",
  "transliteration": "kannada padya",
  "english_translation": "English translation",
  "meaning": "Detailed explanation",
  "themes": ["Theme1", "Theme2"]
}
```

Source: `rakeshmbgit/kagga-website/data/kaggas.ts` → convert using `scripts/convert_kagga_ts.py`.

## Deploy to Railway (Recommended)

### Option A: All-in-One (API + Qdrant on Railway)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial backend"
   git remote add origin https://github.com/YOURUSER/kagga-backend.git
   git push -u origin main
   ```

2. **Create Railway Project**
   - Go to [railway.app](https://railway.app) → New Project
   - "Deploy from GitHub repo" → select your repo
   - Railway detects `Dockerfile` and `railway.json`

3. **Add Qdrant Database**
   - In project: `+ New` → `Database` → `Qdrant`
   - Wait for provisioning (1-2 min)
   - Copy **Internal URL** (e.g., `http://qdrant.railway.internal:6333`)

4. **Set Environment Variables**
   - Go to your API service → Variables
   - Add:
     ```
     OPENAI_API_KEY=sk-your-key
     QDRANT_URL=http://qdrant.railway.internal:6333
     ```

5. **Deploy**
   - Railway auto-deploys on push
   - Get your URL: `https://kagga-api-production-xxxx.railway.app`

6. **Embed Verses (Post-Deploy)**
   - In Railway: API service → Shell
   - Run: `python scripts/embed_verses.py`
   - (One-time, ~30 seconds)

### Option B: Supabase pgvector (No Qdrant)

1. Create Supabase project → enable `pgvector` extension
2. Get connection string: `postgresql://postgres:password@db.xxx.supabase.co:5432/postgres`
3. Modify `vector_search.py` to use `psycopg2` + `pgvector` instead of Qdrant client
4. Set `QDRANT_URL` to Supabase URL (or new env var)

## Run Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ask` | RAG Q&A (question, language, top_k) |
| POST | `/search` | Vector search (query, themes, top_k) |

## Cost Estimate (Monthly)

| Item | Cost |
|------|------|
| Railway (API + Qdrant) | $0–5 (within $5 free credit) |
| OpenAI Embeddings (one-time) | ~$0.01 |
| OpenAI LLM (1000 queries) | ~$0.15 |
| **Total** | **~$0.20–5/mo** |

## Architecture

```
iOS App (SwiftUI)
    │ HTTPS
    ▼
Railway: FastAPI (uvicorn)
    │
    ├─▶ Qdrant (vector search)
    └─▶ OpenAI API
         ├─▶ text-embedding-3-small (embeddings)
         └─▶ gpt-4o-mini (generation)
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OPENAI_API_KEY not set` | Add to Railway Variables |
| Qdrant connection refused | Use **internal** URL, not public |
| Embedding script OOM | Reduce `batch_size` in `embed_verses.py` |
| Slow first query | Qdrant cold start; subsequent queries fast |
| CORS errors | `allow_origins=["*"]` already in `main.py` |