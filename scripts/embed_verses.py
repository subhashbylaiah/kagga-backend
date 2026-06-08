import json
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "kagga_verses"
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "kaggas.json"

if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL)
openai = AsyncOpenAI(api_key=OPENAI_KEY)


async def embed_text(text: str) -> list[float]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def load_verses() -> list[dict]:
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Expected format: data/kaggas.json with array of verses")
        sys.exit(1)
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} verses")
    return data


def create_collection():
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION_NAME in names:
        print(f"Collection '{COLLECTION_NAME}' exists, recreating...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'")


async def main():
    verses = load_verses()
    create_collection()

    points = []
    for v in tqdm(verses, desc="Embedding verses"):
        text = f"{v['kannada_text']}\n{v['transliteration']}\n{v['english_translation']}\n{v['meaning']}"
        vector = await embed_text(text)

        points.append(
            PointStruct(
                id=v["id"],
                vector=vector,
                payload={
                    "verse_number": v["id"],
                    "kannada_text": v["kannada_text"],
                    "transliteration": v["transliteration"],
                    "english_translation": v["english_translation"],
                    "meaning": v["meaning"],
                    "themes": v.get("themes", []),
                },
            )
        )

    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"Done! Upserted {len(points)} verses to Qdrant")


if __name__ == "__main__":
    asyncio.run(main())