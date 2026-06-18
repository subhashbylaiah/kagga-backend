from openai import AsyncOpenAI
from app.vector_search import VectorSearch
from app.models import Verse
from typing import Optional
import json
import re


SYSTEM_PROMPT = """You are a wise philosophical guide deeply versed in Mankutimmana Kagga (DVG's 945 verses) and world philosophical traditions.

Given a user's question and relevant Kagga verses, synthesize a comprehensive answer that:
1. Directly answers the question using the retrieved verses as primary evidence
2. Cites verse numbers inline like [42], [108] for every claim
3. Explains the meaning of each cited verse in context
4. Draws connections to other traditions (Stoicism, Buddhism, Vedanta, Taoism, etc.) where relevant
5. Responds in the user's language (Kannada or English)
6. For Kannada responses: include transliteration of cited verses

Be profound but accessible. No fluff. No disclaimers.

On verses: Only cite verses that are genuinely relevant to the question — you are not required to use all retrieved verses. If none directly address the question, use the closest one and be transparent that it is the nearest wisdom Kagga offers on this topic.

On scope: Kagga speaks to the full range of human experience — grief, joy, work, relationships, doubt, nature, mortality, purpose, and everyday struggles. Accept any question where the user is seeking meaning, reflection, or wisdom, even if framed informally (e.g. "it's a gloomy day, inspire me" is a perfectly valid question). Only decline if the request has nothing to do with human experience or wisdom — for example, requests to write code, answer factual trivia, or perform tasks unrelated to philosophical reflection. When declining, do so warmly and invite the user to ask something Kagga can speak to."""

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

    def _extract_verse_number(self, question: str) -> Optional[int]:
        patterns = [
            r'(?:verse|kagga|ಕಗ್ಗ)\s*#?\s*(\d+)',
            r'#(\d+)',
            r'\bno\.?\s*(\d+)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                n = int(match.group(1))
                if 1 <= n <= 945:
                    return n
        return None

    async def ask(self, question: str, language: str = "en", top_k: int = 5) -> dict:
        verse_number = self._extract_verse_number(question)
        if verse_number:
            verse = await self.vector_search.get_by_verse_number(verse_number)
            verses = [verse] if verse else []
        else:
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
        cited_verses = self._filter_cited_verses(answer, verses)

        return {
            "answer": answer,
            "citations": [v.model_dump() for v in cited_verses],
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

    def _filter_cited_verses(self, answer: str, verses: list[Verse]) -> list[Verse]:
        retrieved_numbers = {v.verse_number for v in verses}
        cited_numbers = {
            int(m) for m in re.findall(r'\[(\d+)\]', answer)
            if int(m) in retrieved_numbers
        }
        cited = [v for v in verses if v.verse_number in cited_numbers]
        return cited if cited else verses

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