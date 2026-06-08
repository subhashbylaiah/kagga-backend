import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.vector_search import VectorSearch
from app.models import Verse, SearchResult


@pytest.fixture
def mock_qdrant_client():
    with patch("app.vector_search.QdrantClient") as mock:
        yield mock.return_value


@pytest.fixture
def mock_openai():
    with patch("app.vector_search.AsyncOpenAI") as mock:
        yield mock.return_value


@pytest.mark.asyncio
async def test_search_returns_results(mock_qdrant_client, mock_openai):
    mock_openai.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )

    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.payload = {
        "verse_number": 42,
        "kannada_text": "ಬದುಕು ಜಟಕ ಬಂಡಿ",
        "transliteration": "Baduku jataka bandi",
        "english_translation": "Life is a cart hitched to the yoke",
        "meaning": "Life is a journey with burdens",
        "themes": ["Life", "Journey"],
    }
    mock_qdrant_client.search.return_value = [mock_hit]

    vs = VectorSearch(qdrant_url="http://test:6333")
    results = await vs.search(query="impermanence", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].verse.verse_number == 42
    assert results[0].score == 0.95
    assert "Life" in results[0].verse.themes


@pytest.mark.asyncio
async def test_search_with_theme_filter(mock_qdrant_client, mock_openai):
    mock_openai.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )
    mock_qdrant_client.search.return_value = []

    vs = VectorSearch(qdrant_url="http://test:6333")
    await vs.search(query="test", themes=["Dharma"], top_k=5)

    call_args = mock_qdrant_client.search.call_args
    assert call_args.kwargs["query_filter"] is not None