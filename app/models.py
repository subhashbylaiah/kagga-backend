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