"""Pronunciation fixes applied to the controller's text before text-to-speech.

Kokoro decides between heteronyms from context, and "wind three zero zero" gets
tagged as the verb ("wind a clock"). Kokoro accepts inline IPA overrides in the
form [word](/phonemes/), so force the noun pronunciation.
IPA is written as \\u escapes to keep this file ASCII (02c8 = stress mark, 026a = short i).
"""

from __future__ import annotations

import re

KOKORO_FIXES = {
    "wind": "/w\u02c8\u026and/",
    "winds": "/w\u02c8\u026andz/",
}

_WORD_RE = re.compile(r"\b(" + "|".join(KOKORO_FIXES) + r")\b", re.IGNORECASE)


def for_tts(text: str, model: str) -> str:
    if "kokoro" not in model.lower():
        return text
    return _WORD_RE.sub(lambda m: f"[{m.group(1)}]({KOKORO_FIXES[m.group(1).lower()]})", text)
