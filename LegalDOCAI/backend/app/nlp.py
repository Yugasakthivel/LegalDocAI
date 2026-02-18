import re
import spacy
import asyncio
from typing import Dict, Any, List, Optional
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
    try:
        for ch in text:
            cp = ord(ch)
            if 0x0B80 <= cp <= 0x0BFF:
                return "Tamil"
            if 0x0900 <= cp <= 0x097F:
                return "Hindi"
            if 0x0C80 <= cp <= 0x0CFF:
                return "Kannada"
            if 0x0C00 <= cp <= 0x0C7F:
                return "Telugu"
            if 0x0D00 <= cp <= 0x0D7F:
                return "Malayalam"
        return "English" if re.search(r"[A-Za-z]", text) else "Unknown"
    except Exception:
        return "Unknown"

import httpx
import urllib.parse

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
        
        # Parallelize OpenAI and MyMemory if needed, but for now just optimize OpenAI
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
        
        # Optimized Async MyMemory fallback
        async with httpx.AsyncClient(timeout=8.0) as client:
            lang_map = {"Tamil": "ta", "Hindi": "hi", "Kannada": "kn", "Telugu": "te", "Malayalam": "ml"}
            src = lang_map.get(lang, "auto")
            # Limit total characters for MyMemory to 1500 for speed
            chunks = [text[i:i+500] for i in range(0, min(len(text), 1500), 500)]
            
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
            translated_chunks = await asyncio.gather(*[fetch_chunk(c) for c in chunks])
            return " ".join(translated_chunks)
    except Exception as e:
        print(f"[DEBUG] translate_to_english_async: error {e}")
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

def perform_ner(text: str) -> Dict[str, List[str]]:
    # 1. Try LLM-based extraction first for better accuracy
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        prompt = (
            "Extract all PERSON names and ORGANIZATION names from the text below. "
            "Return a JSON object with keys 'names' (list of strings) and 'organizations' (list of strings). "
            "Do not include generic terms like 'Vendor', 'Buyer', 'Company'. Only specific names.\n\n"
            + text[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content or "{}"
        import json
        data = json.loads(content)
        llm_names = data.get("names", [])
        llm_orgs = data.get("organizations", [])
    except Exception:
        llm_names = []
        llm_orgs = []

    # 2. Fallback/Augment with spaCy
    nlp = _get_nlp()
    doc = nlp(text)
    spacy_names = []
    spacy_orgs = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            spacy_names.append(ent.text)
        elif ent.label_ == "ORG":
            spacy_orgs.append(ent.text)
            
    # Combine results
    final_names = list(set(llm_names + spacy_names))
    final_orgs = list(set(llm_orgs + spacy_orgs))
    
    # Simple cleaning
    final_names = [n for n in final_names if len(n.strip()) > 2]
    final_orgs = [n for n in final_orgs if len(n.strip()) > 2]

    return {"names": final_names, "organizations": final_orgs}

async def perform_ner_async(text: str) -> Dict[str, List[str]]:
    # 1. Try LLM-based extraction first
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        # Reduced text limit for speed
        prompt = (
            "Extract all PERSON names and ORGANIZATION names from the text below. "
            "Return a JSON object with keys 'names' (list of strings) and 'organizations' (list of strings). "
            "Do not include generic terms like 'Vendor', 'Buyer', 'Company'. Only specific names.\n\n"
            + text[:2500]
        )
        resp = await get_async_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content or "{}"
        import json
        data = json.loads(content)
        llm_names = data.get("names", [])
        llm_orgs = data.get("organizations", [])
    except Exception:
        llm_names = []
        llm_orgs = []

    # 2. Fallback with spaCy (Wrapped in thread for performance)
    def _spacy_ner():
        nlp = _get_nlp()
        if nlp:
            # Limit text for spaCy too
            doc = nlp(text[:5000])
            return [ent.text for ent in doc.ents if ent.label_ == "PERSON"], \
                   [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return [], []

    spacy_names, spacy_orgs = await asyncio.to_thread(_spacy_ner)
            
    final_names = list(set(llm_names + spacy_names))
    final_orgs = list(set(llm_orgs + spacy_orgs))
    return {
        "names": [n for n in final_names if len(n.strip()) > 2],
        "organizations": [n for n in final_orgs if len(n.strip()) > 2]
    }

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

def legal_summarize(text: str) -> str:
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        prompt = "Summarize the following legal text in 3-4 sentences focusing on obligations, risks, and parties:\n\n" + text[:4000]
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        nlp = _get_nlp()
        if nlp:
            doc = nlp(text[:2000])
            sents = [s.text for s in doc.sents]
            return " ".join(sents[:3])
        return "Summary unavailable."

async def legal_summarize_async(text: str) -> str:
    try:
        if not is_openai_ready(): raise RuntimeError("OpenAI not ready")
        # Limit text for speed
        prompt = "Summarize the following legal text in 3-4 sentences focusing on obligations, risks, and parties:\n\n" + text[:2500]
        resp = await get_async_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=250
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Fallback to sync version's fallback wrapped in thread
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
