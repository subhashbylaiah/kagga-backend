from pydantic import BaseModel, Field
from typing import Optional


class Verse(BaseModel):
    verse_number: int = Field(..., ge=1, le=945)
    kannada_text: str
    transliteration: str
    english_translation: str
    meaning: str
    themes: list[str] = []


class SearchResult(BaseModel):
    verse: Verse
    score: float = Field(..., ge=0.0, le=1.0)


class TopicNode(BaseModel):
    id: str
    label: str
    description: str = ""
    verse_count: int
    children: Optional[list["TopicNode"]] = None
    verse_numbers: Optional[list[int]] = None


TopicNode.model_rebuild()