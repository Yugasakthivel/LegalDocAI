import re
import spacy
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from backend.app.core.deps import get_openai_client, get_async_openai_client, is_openai_ready
from backend.app.core.config import OPENAI_MODEL

_nlp: Optional[spacy.language.Language] = None

def _get_nlp() -> Optional[spacy.language.Language]:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[DEBUG] Failed to load spaCy model: {e}")
            _nlp = None
    return _nlp

def detect_language(text: str) -> str:
    lang, _ = detect_language_with_confidence(text)
    return lang

def detect_language_with_confidence(text: str) -> Tuple[str, float]:
    """
    Detects language and returns a confidence score based on script density.
    Handles mixed languages by returning the dominant script.
    """
    if not text or not text.strip():
        return "Unknown", 0.0
    
    counts = {
        "Tamil": 0,
        "Hindi": 0,
        "Kannada": 0,
        "Telugu": 0,
        "Malayalam": 0,
        "English": 0
    }
    
    total_relevant = 0
    for ch in text:
        cp = ord(ch)
        if 0x0B80 <= cp <= 0x0BFF:
            counts["Tamil"] += 1
            total_relevant += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["Hindi"] += 1
            total_relevant += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["Kannada"] += 1
            total_relevant += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["Telugu"] += 1
            total_relevant += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["Malayalam"] += 1
            total_relevant += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["English"] += 1
            total_relevant += 1
            
    if total_relevant == 0:
        return "Unknown", 0.0
        
    dominant_lang = max(counts, key=counts.get)
    confidence = counts[dominant_lang] / total_relevant
    
    return dominant_lang, confidence

def sentence_safe_chunk(text: str, max_chars: int = 1500) -> List[str]:
    """
    Splits text into chunks at sentence boundaries to preserve context.
    """
    if not text:
        return []
    
    # Simple regex for sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If a single sentence is longer than max_chars, split it by character
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i+max_chars].strip())
                current_chunk = ""
            else:
                current_chunk = sentence
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

import httpx
import urllib.parse
from backend.app.services.translation_service import translation_service
from backend.app.services.ner_service import ner_service
from backend.app.services.summarization_service import summarization_service

def translate_to_english_mymemory(text: str, src_lang: str = "auto") -> str:
    """Fallback translation using MyMemory API (free, no key required for low volume)."""
    try:
        print(f"[DEBUG] translate_to_english_mymemory: translating {len(text)} chars from {src_lang}")
        # MyMemory uses 'ta' for Tamil, etc. 
        lang_map = {"Tamil": "ta", "Hindi": "hi", "Kannada": "kn", "Telugu": "te", "Malayalam": "ml"}
        src = lang_map.get(src_lang, "auto")
        
        # Split text into chunks if too long (MyMemory limit is ~500 chars per request)
        chunks = [text[i:i+500] for i in range(0, min(len(text), 2000), 500)]
        translated_chunks = []
        
        with httpx.Client(timeout=10.0) as client:
            for chunk in chunks:
                quoted_text = urllib.parse.quote(chunk)
                url = f"https://api.mymemory.translated.net/get?q={quoted_text}&langpair={src}|en"
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    chunk_trans = data.get("responseData", {}).get("translatedText", "")
                    if chunk_trans:
                        translated_chunks.append(chunk_trans)
                else:
                    print(f"[DEBUG] MyMemory chunk error: {resp.status_code}")
                    translated_chunks.append(chunk) # Fallback to original for this chunk
        
        res = " ".join(translated_chunks)
        print(f"[DEBUG] translate_to_english_mymemory: success, len={len(res)}")
        return res
    except Exception as e:
        print(f"[DEBUG] translate_to_english_mymemory: error {e}")
        return text

def translate_to_english(text: str, src_lang: Optional[str] = None) -> str:
    try:
        lang = src_lang or detect_language(text)
        print(f"[DEBUG] translate_to_english: lang={lang}, text_len={len(text)}")
        if lang == "English" or not text.strip():
            return text
        
        # Try OpenAI first
        if is_openai_ready():
            try:
                # Enhanced prompt to correct errors AND translate
                prompt = (
                    f"The following text is in {lang} and was extracted via OCR. It likely contains spelling mistakes and OCR errors. "
                    "Please correct all spelling and grammatical errors, then translate the text into clear, professional legal English. "
                    "Ensure the final output is free of spelling mistakes. "
                    "Preserve key legal terms and names accurately.\n\n" + text[:4000]
                )
                resp = get_openai_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )
                translated = (resp.choices[0].message.content or "").strip()
                print(f"[DEBUG] translate_to_english: OpenAI success, translated_len={len(translated)}")
                return translated
            except Exception as e:
                print(f"[DEBUG] translate_to_english: OpenAI error, falling back: {e}")
        
        # Fallback to MyMemory if OpenAI fails or not ready
        print("[DEBUG] translate_to_english: Using MyMemory fallback")
        return translate_to_english_mymemory(text, lang)
    except Exception as e:
        print(f"[DEBUG] translate_to_english: Global Error: {e}")
        return text

async def translate_to_english_async(text: str, src_lang: Optional[str] = None) -> str:
    try:
        lang = src_lang or detect_language(text)
        if lang == "English" or not text.strip():
            return text
        
        # 1. OpenAI Path (Cloud, highest quality)
        if is_openai_ready():
            try:
                # Limit input text to 3000 chars for speed in OpenAI
                prompt = (
                    f"The following text is in {lang} and was extracted via OCR. It likely contains spelling mistakes and OCR errors. "
                    "Please correct all spelling and grammatical errors, then translate the text into clear, professional legal English. "
                    "Ensure the final output is free of spelling mistakes. "
                    "Preserve key legal terms and names accurately.\n\n" + text[:3000]
                )
                resp = await get_async_openai_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"[DEBUG] translate_to_english_async: OpenAI error: {e}")
        
        # 2. Local MarianMT Path (Local, mid quality, private)
        # Use sentence-safe chunking for transformer models
        chunks = sentence_safe_chunk(text, max_chars=1000)
        try:
            translated_chunks = await translation_service.translate_batch(chunks, lang, "en")
            if translated_chunks and any(c != original for c, original in zip(translated_chunks, chunks)):
                return " ".join(translated_chunks)
        except Exception as e:
            print(f"[DEBUG] translate_to_english_async: Local translation failed, trying MyMemory: {e}")

        # 3. MyMemory Path (Cloud Fallback, free)
        async with httpx.AsyncClient(timeout=8.0) as client:
            lang_map = {"Tamil": "ta", "Hindi": "hi", "Kannada": "kn", "Telugu": "te", "Malayalam": "ml"}
            src = lang_map.get(lang, "auto")
            
            async def fetch_chunk(chunk):
                try:
                    quoted_text = urllib.parse.quote(chunk)
                    url = f"https://api.mymemory.translated.net/get?q={quoted_text}&langpair={src}|en"
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("responseData", {}).get("translatedText", chunk)
                    return chunk
                except Exception:
                    return chunk

            # Run MyMemory chunks in parallel
            translated_chunks = await asyncio.gather(*[fetch_chunk(c) for c in chunks[:10]]) # Limit to 10 chunks
            return " ".join(translated_chunks)
    except Exception as e:
        print(f"[DEBUG] translate_to_english_async: Global error {e}")
        return text

def correct_ocr_errors(text: str) -> str:
    """Uses LLM to correct OCR errors and spelling mistakes."""
    try:
        if not text.strip(): return text
        if not is_openai_ready(): return text
        prompt = (
            "The following text was extracted via OCR and contains spelling mistakes and noise. "
            "Please rewrite the text into proper, readable English, correcting all spelling and grammatical errors. "
            "Fix names and legal terms where obvious context exists. "
            "Do NOT summarize, just correct the text to be error-free.\n\n" + text[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text

async def correct_ocr_errors_async(text: str) -> str:
    try:
        if not text.strip(): return text
        if not is_openai_ready(): return text
        # Limit input text for speed
        prompt = (
            "The following text was extracted via OCR and contains spelling mistakes and noise. "
            "Please rewrite the text into proper, readable English, correcting all spelling and grammatical errors. "
            "Fix names and legal terms where obvious context exists. "
            "Do NOT summarize, just correct the text to be error-free.\n\n" + text[:3000]
        )
        resp = await get_async_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text

def perform_ner(text: str) -> Dict[str, Any]:
    # 1. Try LLM-based extraction first for high-level names/orgs
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        prompt = (
            "Extract all PERSON names, ORGANIZATION names, DATEs, and MONETARY AMOUNTs from the legal text below. "
            "Return a JSON object with keys: 'names', 'organizations', 'dates', 'money'. "
            "For 'money', include the full string like 'Rs. 50,000'. "
            "Do not include generic terms. Only specific entities.\n\n"
            + text[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content or "{}"
        import json
        llm_data = json.loads(content)
    except Exception:
        llm_data = {"names": [], "organizations": [], "dates": [], "money": []}

    # 2. Local Legal-BERT NER (Domain Specific)
    local_ner = ner_service.extract_entities(text)
    bert_entities = local_ner.get("entities", [])
    
    # 3. spaCy Fallback (General entities)
    nlp = _get_nlp()
    spacy_names = []
    spacy_orgs = []
    spacy_dates = []
    spacy_money = []
    if nlp:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            if ent.label_ == "PERSON": spacy_names.append(ent.text)
            elif ent.label_ == "ORG": spacy_orgs.append(ent.text)
            elif ent.label_ == "DATE": spacy_dates.append(ent.text)
            elif ent.label_ == "MONEY": spacy_money.append(ent.text)

    # 4. Hybrid Fusion & Normalization
    final_names = list(set(llm_data.get("names", []) + spacy_names + [e["text"] for e in bert_entities if e["label"] == "PERSON"]))
    final_orgs = list(set(llm_data.get("organizations", []) + spacy_orgs + [e["text"] for e in bert_entities if e["label"] == "ORG"]))
    
    raw_dates = list(set(llm_data.get("dates", []) + spacy_dates + [e["text"] for e in bert_entities if e["label"] == "DATE"]))
    raw_money = list(set(llm_data.get("money", []) + spacy_money + [e["text"] for e in bert_entities if "MONEY" in e["label"] or "AMOUNT" in e["label"]]))

    # Normalization
    normalized_dates = []
    for d in raw_dates:
        norm = ner_service.normalize_date(d)
        if norm: normalized_dates.append({"original": d, "normalized": norm})
    
    normalized_money = []
    for m in raw_money:
        norm = ner_service.normalize_money(m)
        if norm: normalized_money.append({"original": m, "normalized": norm})

    return {
        "names": [n for n in final_names if len(n.strip()) > 2],
        "organizations": [n for n in final_orgs if len(n.strip()) > 2],
        "dates": normalized_dates,
        "money": normalized_money,
        "metadata": {
            "model": local_ner.get("model", "spacy/llm"),
            "version": local_ner.get("version", "1.0.0")
        }
    }

async def perform_ner_async(text: str) -> Dict[str, Any]:
    # Wrap sync version for now, or implement async specific logic
    return await asyncio.to_thread(perform_ner, text)

def extract_clauses(text: str) -> List[str]:
    kws = [
        "termination", "confidentiality", "liability", "warranty",
        "dispute", "governing law", "payment", "obligation",
        "indemnity", "agreement"
    ]
    found = []
    lower = text.lower()
    for kw in kws:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            found.append(kw)
    return list(dict.fromkeys(found))

def legal_summarize(text: str) -> Dict[str, Any]:
    # 1. Try OpenAI-based extraction first for best abstractive quality
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        prompt = (
            "Summarize the following legal text in 3-4 sentences focusing on obligations, risks, and parties. "
            "Ensure numerical consistency and preserve party names accurately. "
            "Return a JSON object with keys: 'summary' (string), 'key_points' (list of strings).\n\n"
            + text[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "summary": data.get("summary", ""),
            "key_points": data.get("key_points", []),
            "method": "OpenAI",
            "metadata": {"model": OPENAI_MODEL}
        }
    except Exception as e:
        print(f"[DEBUG] OpenAI Summarization failed: {e}")

    # 2. Fallback to Local Summarization Service (Map-Reduce / TextRank)
    res = summarization_service.summarize_chunked(text)
    return {
        "summary": res["summary"],
        "key_points": [], # Local model is purely abstractive/extractive
        "method": res["method"],
        "metadata": res.get("metadata", {})
    }

async def legal_summarize_async(text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(legal_summarize, text)

def process_text_pipeline(text: str) -> Dict[str, Any]:
    language = detect_language(text)
    
    # If English, try to correct OCR errors first
    if language == "English":
        processed_text = correct_ocr_errors(text)
    else:
        # If not English, translate (which now includes correction)
        processed_text = translate_to_english(text, language)
        
    ner = perform_ner(processed_text)
    clauses = extract_clauses(processed_text)
    summary = legal_summarize(processed_text)
    return {
        "language": language,
        "translated_text": processed_text,
        "entities": ner,
        "clauses": clauses,
        "summary": summary,
    }
