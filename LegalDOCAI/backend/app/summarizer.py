from typing import Optional
from .config import SUMMARIZATION_MODEL
import math

# Try to import transformers summarization pipeline
try:
    from transformers import pipeline
    _summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL, device=-1)
except Exception:
    _summarizer = None


def summarize_with_transformers(text: str, max_length: int = 200, min_length: int = 30):
    if not _summarizer:
        raise RuntimeError("Transformers summarizer not available")
    # transformers expects a shorter input sometimes; we chunk if needed
    # naive chunking by characters:
    if len(text) <= 1000:
        return _summarizer(text, max_length=max_length, min_length=min_length)[0]["summary_text"]
    else:
        # chunk by approx 800-1000 chars
        chunks = []
        start = 0
        chunk_size = 900
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size
        summaries = [_summarizer(c, max_length=max_length, min_length=min_length)[0]["summary_text"] for c in chunks]
        return " ".join(summaries)


# Fallback simple summarizer: extract top sentences by length + keyword presence
def fallback_summarize(text: str, max_sentences: int = 5) -> str:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return ""
    # score sentences by length and presence of legal keywords
    keywords = ["plaintiff", "defendant", "judgment", "court", "order", "section", "contract", "agreement", "claim"]
    scored = []
    for s in sentences:
        score = len(s.split())
        s_lower = s.lower()
        for k in keywords:
            if k in s_lower:
                score += 50
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:max_sentences]]
    return " ".join(top)


def summarize_text(text: str, prefer_transformers: bool = True) -> str:
    if prefer_transformers and _summarizer:
        try:
            return summarize_with_transformers(text)
        except Exception:
            return fallback_summarize(text)
    else:
        return fallback_summarize(text)
