from typing import Dict, Any, List, Optional
import re
import spacy
from backend.app.core.config import client as OPENAI_CLIENT, OPENAI_MODEL

_nlp: Optional[spacy.language.Language] = None

def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
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

def translate_to_english(text: str, src_lang: Optional[str] = None) -> str:
    try:
        lang = src_lang or detect_language(text)
        if lang == "English" or not text.strip():
            return text
        
        # Enhanced prompt to correct errors AND translate
        prompt = (
            f"The following text is in {lang} and was extracted via OCR. It likely contains spelling mistakes and OCR errors. "
            "Please correct all spelling and grammatical errors, then translate the text into clear, professional legal English. "
            "Ensure the final output is free of spelling mistakes. "
            "Preserve key legal terms and names accurately.\n\n" + text[:4000]
        )
        resp = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text

def correct_ocr_errors(text: str) -> str:
    """Uses LLM to correct OCR errors and spelling mistakes."""
    try:
        if not text.strip(): return text
        prompt = (
            "The following text was extracted via OCR and contains spelling mistakes and noise. "
            "Please rewrite the text into proper, readable English, correcting all spelling and grammatical errors. "
            "Fix names and legal terms where obvious context exists. "
            "Do NOT summarize, just correct the text to be error-free.\n\n" + text[:4000]
        )
        resp = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text

def perform_ner(text: str) -> Dict[str, List[str]]:
    # 1. Try LLM-based extraction first for better accuracy
    try:
        prompt = (
            "Extract all PERSON names and ORGANIZATION names from the text below. "
            "Return a JSON object with keys 'names' (list of strings) and 'organizations' (list of strings). "
            "Do not include generic terms like 'Vendor', 'Buyer', 'Company'. Only specific names.\n\n"
            + text[:4000]
        )
        resp = OPENAI_CLIENT.chat.completions.create(
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
        prompt = "Summarize the following legal text in 3-4 sentences focusing on obligations, risks, and parties:\n\n" + text[:4000]
        resp = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        nlp = _get_nlp()
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        return " ".join(sents[:3]) if sents else ""

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
