import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["QDRANT_URL"] = "http://test:6333"

import importlib
import scripts.embed_verses as embed_module
importlib.reload(embed_module)

from scripts.embed_verses import load_verses, create_collection, embed_text


class TestEmbedVerses:
    @pytest.fixture
    def sample_verses(self):
        return [
            {
                "id": 1,
                "kannada_text": "ಮಂಕುತಿಂಮ ಕಗ್ಗ",
                "transliteration": "Mankutimma kagga",
                "english_translation": "Dull Thimma's rigmarole",
                "meaning": "Opening verse about humility",
                "themes": ["Humility", "Beginning"]
            },
            {
                "id": 42,
                "kannada_text": "ಬದುಕು ಜಟಕ ಬಂಡಿ",
                "transliteration": "Baduku jataka bandi",
                "english_translation": "Life is a cart hitched to the yoke",
                "meaning": "Life is a journey with burdens",
                "themes": ["Life", "Journey", "Impermanence"]
            }
        ]

    @pytest.fixture
    def mock_data_file(self, tmp_path, sample_verses):
        data_file = tmp_path / "kaggas.json"
        data_file.write_text(json.dumps(sample_verses))
        return data_file

    def test_load_verses(self, mock_data_file, monkeypatch):
        monkeypatch.setattr(embed_module, "DATA_PATH", mock_data_file)

        verses = load_verses()

        assert len(verses) == 2
        assert verses[0]["id"] == 1
        assert verses[1]["id"] == 42
        assert "kannada_text" in verses[0]

    def test_load_verses_missing_file(self, monkeypatch, tmp_path):
        missing_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(embed_module, "DATA_PATH", missing_file)

        with pytest.raises(SystemExit):
            load_verses()

    @patch.object(embed_module, "client", new_callable=MagicMock)
    def test_create_collection_new(self, mock_client):
        mock_client.get_collections.return_value = MagicMock(collections=[])

        create_collection()

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "kagga_verses"
        assert call_args.kwargs["vectors_config"].size == 1536

    @patch.object(embed_module, "client", new_callable=MagicMock)
    def test_create_collection_existing(self, mock_client):
        existing = MagicMock()
        existing.name = "kagga_verses"
        mock_client.get_collections.return_value = MagicMock(collections=[existing])

        create_collection()

        mock_client.delete_collection.assert_called_once_with("kagga_verses")
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(embed_module, "openai", new_callable=AsyncMock)
    async def test_embed_text(self, mock_openai):
        mock_openai.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        )

        result = await embed_text("test text")

        assert len(result) == 1536
        mock_openai.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )