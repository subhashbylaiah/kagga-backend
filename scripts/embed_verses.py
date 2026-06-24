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
DATA_PATH = Path(__file__).parent.parent / "data" / "kaggas.json"

if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL)
openai = AsyncOpenAI(api_key=OPENAI_KEY)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in response.data]


def load_verses() -> list[dict]:
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
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
        vectors_config={
            "english": VectorParams(size=1536, distance=Distance.COSINE),
            "kannada": VectorParams(size=1536, distance=Distance.COSINE),
        },
    )
    print(f"Created collection '{COLLECTION_NAME}' with named vectors")


async def main():
    verses = load_verses()
    create_collection()

    batch_size = 100
    points = []

    for i in tqdm(range(0, len(verses), batch_size), desc="Embedding batches"):
        batch = verses[i:i+batch_size]

        english_texts = [
            f"{v['english_translation']}\n{v['meaning']}"
            for v in batch
        ]
        kannada_texts = [
            f"{v['kannada_text']}\n{v['transliteration']}"
            for v in batch
        ]

        english_vectors = await embed_batch(english_texts)
        kannada_vectors = await embed_batch(kannada_texts)

        for v, en_vec, kn_vec in zip(batch, english_vectors, kannada_vectors):
            points.append(
                PointStruct(
                    id=v["id"],
                    vector={
                        "english": en_vec,
                        "kannada": kn_vec,
                    },
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

    for i in tqdm(range(0, len(points), 100), desc="Upserting to Qdrant"):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])

    print(f"Done! Upserted {len(points)} verses to Qdrant")


if __name__ == "__main__":
    asyncio.run(main())
