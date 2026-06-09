from openai import AsyncOpenAI
from app.vector_search import VectorSearch
from app.models import Verse
from typing import Optional
import json


SYSTEM_PROMPT = """You are a wise philosophical guide deeply versed in Mankutimmana Kagga (DVG's 945 verses) and world philosophical traditions.

Given a user's question and relevant Kagga verses, synthesize a comprehensive answer that:
1. Directly answers the question using the retrieved verses as primary evidence
2. Cites verse numbers inline like [42], [108] for every claim
3. Explains the meaning of each cited verse in context
4. Draws connections to other traditions (Stoicism, Buddhism, Vedanta, Taoism, etc.) where relevant
5. Responds in the user's language (Kannada or English)
6. For Kannada responses: include transliteration of cited verses

Be profound but accessible. No fluff. No disclaimers."""

FOLLOWUP_PROMPT = """Based on this Kagga answer, suggest 3 short follow-up questions the user might naturally want to ask next.
Return only a JSON array of 3 strings, nothing else. Each question should be concise (under 10 words).
Example: ["What does Kagga say about grief?", "How to practice detachment daily?", "Tell me more about verse 42"]"""


USER_PROMPT_TEMPLATE = """Question: {question}
Language: {language}

Retrieved Kagga Verses:
{verses}

Provide a comprehensive answer with:
- Direct answer with verse citations [number]
- Explanation of each verse's relevance
- Cross-references to other philosophical traditions
- Practical insight for daily life"""


class RAGPipeline:
    def __init__(self, vector_search: VectorSearch, openai_key: str, model: str = "gpt-4o-mini"):
        self.vector_search = vector_search
        self.openai = AsyncOpenAI(api_key=openai_key)
        self.model = model

    async def ask(self, question: str, language: str = "en", top_k: int = 5) -> dict:
        search_results = await self.vector_search.search(query=question, top_k=top_k)
        verses = [r.verse for r in search_results]

        verses_text = self._format_verses(verses, language)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            language="Kannada" if language == "kn" else "English",
            verses=verses_text,
        )

        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content or ""

        cross_refs = self._extract_cross_references(answer)
        suggested_questions = await self._generate_followups(question, answer, language)

        return {
            "answer": answer,
            "citations": [v.model_dump() for v in verses],
            "cross_references": cross_refs,
            "suggested_questions": suggested_questions,
        }

    def _format_verses(self, verses: list[Verse], language: str) -> str:
        lines = []
        for v in verses:
            if language == "kn":
                lines.append(
                    f"[{v.verse_number}] {v.kannada_text}\n"
                    f"    Transliteration: {v.transliteration}\n"
                    f"    English: {v.english_translation}\n"
                    f"    Meaning: {v.meaning}\n"
                    f"    Themes: {', '.join(v.themes)}"
                )
            else:
                lines.append(
                    f"[{v.verse_number}] {v.english_translation}\n"
                    f"    Kannada: {v.kannada_text}\n"
                    f"    Transliteration: {v.transliteration}\n"
                    f"    Meaning: {v.meaning}\n"
                    f"    Themes: {', '.join(v.themes)}"
                )
        return "\n\n".join(lines)

    async def _generate_followups(self, question: str, answer: str, language: str) -> list[str]:
        try:
            lang = "Kannada" if language == "kn" else "English"
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"Question: {question}\n\nAnswer summary: {answer[:500]}\n\nLanguage: {lang}\n\n{FOLLOWUP_PROMPT}"},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            content = response.choices[0].message.content or "[]"
            return json.loads(content)
        except Exception:
            return []

    def _extract_cross_references(self, answer: str) -> list[dict]:
        traditions = [
            "Stoicism", "Buddhism", "Vedanta", "Advaita", "Taoism",
            "Jainism", "Sikhism", "Christianity", "Islam", "Sufism",
            "Existentialism", "Pragmatism", "Confucianism",
        ]
        found = []
        for t in traditions:
            if t.lower() in answer.lower():
                found.append({"tradition": t, "mentioned": True})
        return found