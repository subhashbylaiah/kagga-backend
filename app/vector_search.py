from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, SearchParams
from openai import AsyncOpenAI
from typing import Optional
from app.models import Verse, SearchResult


class VectorSearch:
    def __init__(self, qdrant_url: str, collection_name: str = "kagga_verses"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.openai = AsyncOpenAI()

    async def search(
        self,
        query: str,
        themes: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        query_vector = await self._embed(query)

        qdrant_filter = None
        if themes:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="themes",
                        match=MatchAny(any=themes),
                    )
                ]
            )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128),
            with_payload=True,
        )

        return [
            SearchResult(
                verse=Verse(
                    verse_number=hit.payload["verse_number"],
                    kannada_text=hit.payload["kannada_text"],
                    transliteration=hit.payload["transliteration"],
                    english_translation=hit.payload["english_translation"],
                    meaning=hit.payload["meaning"],
                    themes=hit.payload.get("themes", []),
                ),
                score=hit.score,
            )
            for hit in hits
        ]

    async def get_by_verse_number(self, verse_number: int) -> Optional[Verse]:
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="verse_number",
                        match=MatchValue(value=verse_number),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            return None
        payload = results[0].payload
        return Verse(
            verse_number=payload["verse_number"],
            kannada_text=payload["kannada_text"],
            transliteration=payload["transliteration"],
            english_translation=payload["english_translation"],
            meaning=payload["meaning"],
            themes=payload.get("themes", []),
        )

    async def _embed(self, text: str) -> list[float]:
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding