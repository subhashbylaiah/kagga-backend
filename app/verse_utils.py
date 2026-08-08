import re
from typing import Optional

VERSE_NUMBER_PATTERNS = [
    r'(?:verse|kagga|ಕಗ್ಗ)\s*#?\s*(\d+)',
    r'#(\d+)',
    r'\bno\.?\s*(\d+)\b',
]


def extract_verse_number(question: str) -> Optional[int]:
    for pattern in VERSE_NUMBER_PATTERNS:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            if 1 <= n <= 945:
                return n
    return None
