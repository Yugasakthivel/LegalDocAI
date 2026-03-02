import re
import time
import json
import random
import math
import hashlib
from collections import Counter
from typing import List, Dict, Any, Optional
import dateparser
from backend.app.core.config import OPENAI_MODEL
from backend.app.core.deps import get_async_openai_client, is_openai_ready, mark_quota_exhausted
from backend.app.nlp import (
    _get_nlp, translate_to_english, translate_to_english_async, 
    correct_ocr_errors, correct_ocr_errors_async,
    detect_language, perform_ner as shared_perform_ner, perform_ner_async
)
import asyncio
from backend.app.services.ml_service import classifier
from backend.app.services.compliance_service import jurisdiction_checks
from backend.app.services.timeline_service import timeline_service
from backend.app.services.clause_presence_service import clause_presence_service
from backend.app.services.compliance_checker import compliance_service
from backend.app.services.comparison_service import comparison_service
from backend.app.services.rag_service import legal_rag_service
from backend.app.services.report_service import report_service
from typing import Tuple
import httpx

# Constants
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"\+?\d[\d\s-]{7,}\d"
DATE_PATTERN = r"(?i)\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b"
CLAUSE_KEYWORDS = [
    "termination", "confidentiality", "liability", "warranty",
    "dispute", "governing law", "payment", "obligation",
    "indemnity", "agreement"
]
LAND_TYPE_PATTERNS = {
    "Nanjai": ["nanjai", "wet land", "wetland"],
    "Punjai": ["punjai", "punja", "dry land", "dryland"],
    "Panchami": ["panchami", "pancham"],
    "Agricultural": ["agricultural", "farmland", "cultivable"],
    "Residential": ["residential", "house site", "housing plot"],
    "Commercial": ["commercial", "shop", "industrial", "business use"],
}
try:
    import phonenumbers
except ImportError:
    phonenumbers = None

# --- Helper Functions ---

_ENTITY_STOPWORDS = {
    "copy", "store", "name", "names", "organization", "organizations",
    "government", "gov", "govt", "india", "document", "summary",
    # role/office words that should not be treated as person names
    "registrar", "register", "sub-registrar", "subregistrar", "office",
    "department", "vendor", "purchaser", "notary", "witness", "stamp",
}
_ORG_HINTS = {
    "ltd", "limited", "llp", "pvt", "private", "inc", "corp", "company",
    "co", "bank", "university", "college", "court", "department", "ministry",
    "authority", "board", "trust", "society", "agency", "institute",
    "registrar", "registry", "office", "sub-registrar", "cooperative", "coop", "co-op",
}

def _normalize_entity_text(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    try:
        s = correct_ocr_errors(s) or s
    except Exception:
        pass
    s = re.sub(r"[^A-Za-z\s\.\-&']", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,-")
    return s

def _looks_like_ocr_noise(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return True
    words = re.findall(r"[A-Za-z]{2,}", s)
    if not words:
        return True
    letters = len(re.findall(r"[A-Za-z]", s))
    if letters < 3:
        return True
    bad_long = sum(1 for w in words if len(w) >= 7 and not re.search(r"[aeiou]", w, flags=re.IGNORECASE))
    if bad_long / max(1, len(words)) > 0.35:
        return True
    avg_len = sum(len(w) for w in words) / max(1, len(words))
    if avg_len > 11.0:
        return True
    if re.search(r"(.)\1\1\1", s):
        return True
    return False

def sanitize_names(raw: List[str]) -> List[str]:
    cleaned = []
    for n in raw:
        s = _normalize_entity_text(n)
        if len(s) < 3 or len(s) > 60: continue
        if re.search(r"\d", s): continue
        if _looks_like_ocr_noise(s): continue
        tokens = [t for t in s.split() if any(ch.isalpha() for ch in t)]
        if not tokens: continue
        if len(tokens) < 2: continue
        if len(tokens) > 6: continue
        if any(t.lower() in _ENTITY_STOPWORDS for t in tokens): continue
        if any(any(h in t.lower() for h in _ORG_HINTS) for t in tokens): continue
        latin_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
        tamil_tokens = [t for t in tokens if re.search(r"[\u0B80-\u0BFF]", t)]
        if latin_tokens and not tamil_tokens:
            if any(t.isupper() and len(t) > 24 for t in latin_tokens): continue
            if any(not (re.match(r"^[A-Z][a-z]+(?:[-'][a-z]+)*$", t) or re.match(r"^[A-Z]\.?$", t) or (t.isupper() and 2 <= len(t) <= 30)) for t in latin_tokens): continue
            if any(len(t) >= 4 and not re.search(r"[aeiouAEIOU]", t) and not re.match(r"^[A-Z]{2,}$", t) for t in latin_tokens): continue
        joined = " ".join(tokens)
        letters = len(re.findall(r"[A-Za-z]", joined))
        total = len(joined)
        if total == 0 or letters/total < 0.7: continue
        initials_only = latin_tokens and all(re.match(r"^[A-Z]\.?$", t) for t in latin_tokens) and len(latin_tokens) >= 2
        if latin_tokens and not initials_only:
            if any(re.search(r"(?i)[bcdfghjklmnpqrstvwxyz]{4,}", t) for t in latin_tokens): continue
            vowels = len(re.findall(r"[aeiouAEIOU]", joined))
            all_caps = all(t.isupper() for t in latin_tokens)
            if not all_caps and (vowels / max(1, letters)) < 0.33: continue
        if len(re.findall(r"[aeiouAEIOU]", joined)) == 0 and not initials_only: continue
        if len(re.findall(r"[^A-Za-z\s\.\-']", joined)) > max(1, total//10): continue
        cleaned.append(joined)
    seen = set()
    out = []
    for x in cleaned:
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)
    return out

def sanitize_orgs(raw: List[str]) -> List[str]:
    cleaned = []
    for n in raw:
        s = _normalize_entity_text(n)
        if len(s) < 3 or len(s) > 80: continue
        if _looks_like_ocr_noise(s): continue
        tokens = [t for t in s.split() if any(ch.isalpha() for ch in t)]
        if not tokens: continue
        if len(tokens) > 8: continue
        has_hint = any((t.lower().strip(".") in _ORG_HINTS) or any(h in t.lower() for h in _ORG_HINTS) for t in tokens)
        if len(tokens) < 2 and not has_hint: continue
        short_ok = {"co", "ltd", "pvt", "llp", "inc", "plc", "gov"}
        if not has_hint and any((len(t) <= 3 and t.lower().strip(".") not in short_ok) for t in tokens):
            continue
        if any("coop" in t.lower() for t in tokens):
            if sum(len(t) for t in tokens) < 10:
                continue
        if all(t.lower() in _ENTITY_STOPWORDS for t in tokens):
            continue
        latin_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
        if latin_tokens:
            if any((t.isupper() and len(t) > 4) for t in latin_tokens): continue
            if any((not t.isupper()) and (not re.match(r"^[A-Z][a-z][A-Za-z\.\-']{0,30}$", t)) for t in latin_tokens if len(t) > 2): continue
            if any(re.search(r"(?i)[bcdfghjklmnpqrstvwxyz]{4,}", t) for t in latin_tokens): continue
        joined = " ".join(tokens)
        letters = len(re.findall(r"[A-Za-z]", joined))
        total = len(joined)
        if total == 0 or letters/total < 0.6: continue
        if latin_tokens:
            vowels = len(re.findall(r"[aeiouAEIOU]", joined))
            all_caps = all(t.isupper() for t in latin_tokens)
            if not all_caps and (vowels / max(1, letters)) < 0.3: continue
        if len(re.findall(r"[^A-Za-z0-9\s\.\-&']", joined)) > max(1, total//10): continue
        cleaned.append(joined)
    seen = set()
    out = []
    for x in cleaned:
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)
    return out

def extract_tamil_names(text: str) -> List[str]:
    out: List[str] = []
    t = (text or "")
    for m in re.finditer(r"([\u0B80-\u0BFF]{2,12}(?:\s+[\u0B80-\u0BFF]{2,12}){1,3})", t):
        s = re.sub(r"\s+", " ", m.group(1)).strip()
        if 3 <= len(s) <= 80:
            out.append(s)
        if len(out) >= 8:
            break
    return out

def transliterate_tamil_to_latin(text: str) -> str:
    t = text or ""
    vowels = {
        "அ": "a","ஆ": "aa","இ": "i","ஈ": "ii","உ": "u","ஊ": "uu","எ": "e","ஏ": "ee","ஐ": "ai","ஒ": "o","ஓ": "oo","ஔ": "au",
    }
    vowel_signs = {
        "ா": "aa","ி": "i","ீ": "ii","ு": "u","ூ": "uu","ெ": "e","ே": "ee","ை": "ai","ொ": "o","ோ": "oo","ௌ": "au",
    }
    cons = {
        "க": "k","ங": "ng","ச": "ch","ஜ": "j","ஞ": "nj","ட": "t","ண": "n","த": "th","ந": "n","ப": "p","ம": "m","ய": "y","ர": "r","ல": "l","வ": "v","ழ": "zh","ள": "l","ற": "r","ன": "n",
    }
    pulli = "்"
    out = []
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if ch in vowels:
            out.append(vowels[ch])
            i += 1
            continue
        if ch in cons:
            base = cons[ch]
            next_ch = t[i+1] if i+1 < n else ""
            if next_ch in vowel_signs:
                out.append(base + vowel_signs[next_ch])
                i += 2
                continue
            if next_ch == pulli:
                out.append(base)
                i += 2
                continue
            out.append(base + "a")
            i += 1
            continue
        if ch in vowel_signs:
            out.append(vowel_signs[ch])
            i += 1
            continue
        i += 1
    s = "".join(out)
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_signatures(text: str) -> List[str]:
    found = []
    for m in re.finditer(r"(digitally\s+signed(?:\s+by)?|signed by|signature|authorized signatory|attested by)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)", text, flags=re.IGNORECASE):
        nm = m.group(2).strip()
        if nm and nm not in found:
            found.append(nm)
    return found

def extract_phones_validated(text: str) -> List[str]:
    out = []
    if phonenumbers:
        try:
            for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
                num = match.number
                if not phonenumbers.is_valid_number(num): continue
                if phonenumbers.region_code_for_number(num) != "IN": continue
                nt = phonenumbers.number_type(num)
                if nt not in (phonenumbers.PhoneNumberType.MOBILE, phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE): continue
                digits = str(num.national_number)
                if len(digits) != 10 or digits[0] not in "6789": continue
                val = f"+91 {digits}"
                if val not in out: out.append(val)
        except Exception as e:
            print(f"[DEBUG] PhoneNumberMatcher failed: {e}")
    if not out:
        for m in re.finditer(r"(\+?\d[\d\-\s\(\)]{6,}\d)", text):
            s = m.group(1)
            raw_digits = re.sub(r"\D", "", s)
            if len(raw_digits) == 12 and raw_digits.startswith("91"):
                digits = raw_digits[-10:]
            else:
                digits = raw_digits
            if len(digits) != 10 or digits[0] not in "6789": continue
            if re.search(DATE_PATTERN, s): continue
            sep_count = len(re.findall(r"[\s\-]", s))
            if sep_count > (len(raw_digits) // 3): continue
            val = digits
            if val not in out: out.append(val)
    return out

def phone_compliance_tag(s: str) -> str:
    try:
        if phonenumbers:
            digits = re.sub(r"\D", "", s or "")
            if len(digits) == 10: num = phonenumbers.parse(f"+91{digits}", "IN")
            else: num = phonenumbers.parse(s, "IN")
            if phonenumbers.is_valid_number(num) and phonenumbers.region_code_for_number(num) == "IN":
                nt = phonenumbers.number_type(num)
                if nt in (phonenumbers.PhoneNumberType.MOBILE, phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE):
                    return "IN Mobile Valid"
            return "Unverified"
        digits = re.sub(r"\D", "", s or "")
        if len(digits) == 10 and digits[0] in "6789": return "IN Mobile Format"
        return "Unverified"
    except Exception: return "Unverified"

def detect_clause_headings(text: str) -> List[str]:
    heads = []
    for line in text.splitlines():
        l = line.strip()
        if not l or len(l) > 120: continue
        if re.match(r"^(\d+(\.\d+)*|\([a-zA-Z]\)|[A-Z][A-Za-z\s]{2,})", l):
            alpha_only = re.sub(r"[^A-Za-z]","",l)
            upper_ratio = sum(1 for c in alpha_only if c.isupper())/max(1,len(alpha_only))
            if upper_ratio > 0.6 or re.match(r"^\d+(\.\d+)*\s", l) or re.match(r"^\([a-zA-Z]\)\s", l):
                if l not in heads: heads.append(l)
    return heads

def extract_structured_layout(text: str) -> List[Dict[str, Any]]:
    """
    Groups text into headings, paragraphs, and potential tables for layout awareness.
    """
    headings = set(detect_clause_headings(text))
    layout = []
    current_block = []
    
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Detect table-like structures (e.g., lines with multiple tabs or large spaces)
        if re.search(r"\s{4,}", line) or line.count("\t") >= 2:
            if current_block:
                layout.append({"type": "paragraph", "content": " ".join(current_block)})
                current_block = []
            
            table_rows = []
            while i < len(lines) and (re.search(r"\s{3,}", lines[i]) or lines[i].count("\t") >= 1 or not lines[i].strip()):
                if lines[i].strip():
                    # Split by 3+ spaces or tab
                    cells = re.split(r"\s{3,}|\t", lines[i].strip())
                    table_rows.append(cells)
                i += 1
            layout.append({"type": "table", "content": table_rows})
            continue

        if line in headings:
            if current_block:
                layout.append({"type": "paragraph", "content": " ".join(current_block)})
                current_block = []
            layout.append({"type": "heading", "content": line})
        else:
            current_block.append(line)
        i += 1
            
    if current_block:
        layout.append({"type": "paragraph", "content": " ".join(current_block)})
        
    return layout

def compute_layout_score(text: str) -> float:
    """
    Computes a numerical layout score based on text structural consistency.
    Checks for balanced paragraphs, heading density, and alignment hints.
    """
    if not text or len(text) < 100:
        return 0.5
        
    lines = text.splitlines()
    total_lines = len(lines)
    
    # 1. Heading Density (Authentic legal docs have structured headings)
    headings = detect_clause_headings(text)
    heading_ratio = len(headings) / max(1, total_lines / 20) # Expect ~1 heading per 20 lines
    h_score = min(1.0, heading_ratio)
    
    # 2. Line Length Consistency (Center-aligned or chaotic text is suspicious)
    line_lengths = [len(l.strip()) for l in lines if len(l.strip()) > 5]
    if not line_lengths:
        return 0.5
    avg_len = sum(line_lengths) / len(line_lengths)
    variance = sum((x - avg_len) ** 2 for x in line_lengths) / len(line_lengths)
    std_dev = variance ** 0.5
    
    # Highly chaotic line lengths (standard dev > 30) suggest poor OCR or layout tampering
    l_score = 1.0 - min(0.5, std_dev / 100)
    
    # 3. Structural Consistency (Tables/Lists)
    structured_blocks = extract_structured_layout(text)
    table_count = sum(1 for b in structured_blocks if b["type"] == "table")
    s_score = min(1.0, 0.5 + (table_count * 0.1))
    
    final_score = (h_score * 0.3) + (l_score * 0.4) + (s_score * 0.3)
    return float(final_score)

def extract_land_classification(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    matched: List[str] = []
    for label, patterns in LAND_TYPE_PATTERNS.items():
        if any(p in t for p in patterns):
            matched.append(label)
    primary = matched[0] if matched else "Unknown"
    return {
        "primary_land_type": primary,
        "land_types_detected": matched,
        "is_property_related": any(k in t for k in ["survey", "patta", "land", "property", "acre", "hectare"])
    }

# --- Analysis Functions ---

# --- Analysis Functions ---

def summarize_fallback(text: str, max_words: int = 120) -> str:
    """
    Summarize text using high-speed local service (Hybrid Map-Reduce/TextRank).
    """
    from backend.app.services.summarization_service import summarization_service
    res = summarization_service.summarize_chunked(text)
    return res["summary"]

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _is_resume_like(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    cues = [
        "objective",
        "professional summary",
        "skills",
        "certifications",
        "projects",
        "internship",
        "education",
        "cgpa",
        "diploma",
        "b.e",
        "b.tech",
        "resume",
    ]
    hits = sum(1 for c in cues if c in t)
    return hits >= 3

def _split_resume_sections(text: str) -> Dict[str, List[str]]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return {}
    header_map = {
        r"objective": "objective",
        r"professional summary|summary": "summary",
        r"skills\s*&\s*abilities|skills\s+and\s+abilities|skills": "skills",
        r"certifications?": "certifications",
        r"projects?": "projects",
        r"internship|intern": "internship",
        r"education|diploma|b\.e|b\.tech|bachelor": "education",
    }
    for pattern, key in header_map.items():
        normalized = re.sub(rf"(?i)\b({pattern})\b", rf"\n{key}:", normalized)
    lines = [l.strip() for l in normalized.split("\n") if l.strip()]
    sections: Dict[str, List[str]] = {}
    current = None
    for line in lines:
        if line.endswith(":"):
            current = line[:-1].strip().lower()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return sections

def _clean_resume_fragment(text: str, max_chars: int = 220) -> str:
    t = _normalize_spaces(text)
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0].strip()
    return t

def summarize_resume(text: str) -> str:
    sections = _split_resume_sections(text or "")
    parts: List[str] = []

    summary_blob = " ".join(sections.get("summary", []) + sections.get("objective", []))
    if summary_blob:
        parts.append(f"Objective: {_clean_resume_fragment(summary_blob, 220)}")

    edu_blob = " ".join(sections.get("education", []))
    if not edu_blob:
        edu_blob = text
    edu_bits = []
    for m in re.finditer(r"(?i)\b(diploma|b\.e|b\.tech|bachelor)\b[^\.]{0,80}", edu_blob):
        edu_bits.append(_clean_resume_fragment(m.group(0), 120))
    cgpa = re.search(r"(?i)\b(cgpa|gpa)\b[^0-9]{0,6}([0-9]\.?[0-9]?)", edu_blob)
    if edu_bits or cgpa:
        edu_text = "; ".join(dict.fromkeys(edu_bits))[:180]
        if cgpa:
            edu_text = f"{edu_text} CGPA {cgpa.group(2)}".strip()
        if edu_text:
            parts.append(f"Education: {edu_text}")

    skills_blob = " ".join(sections.get("skills", []))
    if skills_blob:
        skills = [s.strip() for s in re.split(r"[,/|]", skills_blob) if s.strip()]
        if skills:
            parts.append(f"Skills: {', '.join(skills[:8])}")

    cert_blob = " ".join(sections.get("certifications", []))
    if cert_blob:
        certs = [c.strip() for c in re.split(r"[,/|]", cert_blob) if c.strip()]
        if certs:
            parts.append(f"Certifications: {', '.join(certs[:4])}")

    proj_blob = " ".join(sections.get("projects", []))
    if proj_blob:
        proj_bits = [p.strip() for p in re.split(r"→|;|\|", proj_blob) if p.strip()]
        if proj_bits:
            parts.append(f"Projects: {', '.join(proj_bits[:3])}")

    intern_blob = " ".join(sections.get("internship", []))
    if intern_blob:
        parts.append(f"Internship: {_clean_resume_fragment(intern_blob, 140)}")

    if not parts:
        return _normalize_spaces(text or "")[:320] + "..."

    return " ".join(parts)

def _summary_needs_fix(summary: str) -> bool:
    if not summary:
        return True
    s = summary.strip()
    if len(s) < 40:
        return True
    if len(s) > 500 and s.count(".") < 2:
        return True
    if len(re.findall(r"[A-Za-z]", s)) < 20:
        return True
    # OCR-gibberish guard: too many long tokens without vowels usually means unreadable text.
    words = re.findall(r"[A-Za-z]{2,}", s)
    if len(words) < 8:
        return True
    no_vowel_long = sum(1 for w in words if len(w) >= 7 and not re.search(r"[aeiou]", w, flags=re.IGNORECASE))
    if no_vowel_long / max(1, len(words)) > 0.2:
        return True
    avg_len = sum(len(w) for w in words) / max(1, len(words))
    if avg_len > 9.5:
        return True
    return False

def _extract_readable_sentences(text: str, limit: int = 2) -> List[str]:
    t = _normalize_spaces(text or "")
    if not t:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+|\n+", t) if s and s.strip()]
    out: List[str] = []
    for s in sents:
        if len(s) < 35:
            continue
        if _looks_like_ocr_noise(s):
            continue
        words = re.findall(r"[A-Za-z]{2,}", s)
        if len(words) < 6:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out

def _build_structured_summary(
    doc_type: str,
    names: List[str],
    orgs: List[str],
    dates: List[str],
    clause_counter: Counter,
    source_text: str,
) -> str:
    parts: List[str] = []
    d_type = (doc_type or "").strip()
    if d_type and d_type.lower() not in {"unknown", "other", "other document"}:
        parts.append(f"Document type: {d_type}.")
    if names:
        parts.append(f"Names: {', '.join((names or [])[:3])}.")
    if orgs:
        parts.append(f"Organizations: {', '.join((orgs or [])[:2])}.")
    if dates:
        parts.append(f"Dates: {', '.join((dates or [])[:3])}.")
    try:
        top_clauses = [k for k, v in (clause_counter or {}).most_common(5) if int(v or 0) > 0][:3]
    except Exception:
        top_clauses = []
    if top_clauses:
        parts.append(f"Key clauses detected: {', '.join(top_clauses)}.")
    readable = _extract_readable_sentences(source_text or "", limit=1)
    if readable:
        parts.append(readable[0])
    if not parts:
        return "No clean summary available from OCR text. Please upload a clearer scan."
    return _normalize_spaces(" ".join(parts))

async def translate_text_simple_async(text: str, target_lang: str) -> str:
    try:
        tgt = (target_lang or "English").strip()
        if not (text or "").strip():
            return ""
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": (text or "")[:5000], "langpair": f"en|{tgt[:2].lower()}"},
            )
            data = resp.json()
            out = ((data or {}).get("responseData") or {}).get("translatedText") or ""
            return out or text
    except Exception:
        return text

def answer_with_fallback(question: str, context: str) -> str:
    q = (question or "").strip().lower()
    t = (context or "").strip()
    if not t:
        return "Document has no text content."
    # Targeted Q&A shortcuts
    if any(k in q for k in ["name", "person", "who", "party", "signer"]):
        bad_name_words = {
            "tamil", "nadu", "india", "government", "govt", "office", "department", "district", "state",
            "taluk", "village", "survey", "road", "street", "pincode", "pin", "lic", "license", "licence",
            "number", "no", "stamp", "bond", "ec", "uidai", "passport", "card", "authority"
        }

        def _clean_person_name(x: str) -> str:
            s = re.sub(r"[^A-Za-z\s\.\-']", " ", (x or "")).strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def _is_likely_person_name(x: str) -> bool:
            s = _clean_person_name(x)
            if len(s) < 3 or len(s) > 80:
                return False
            if re.search(r"\d", s):
                return False
            toks = [tok.strip(" .") for tok in s.split() if tok.strip(" .")]
            if not toks:
                return False
            lowered = [tok.lower() for tok in toks]
            if any(tok in bad_name_words for tok in lowered):
                return False
            # Prefer proper person-like names: at least 2 clean tokens.
            alpha_toks = [tok for tok in toks if re.search(r"[A-Za-z]", tok)]
            if len(alpha_toks) < 2:
                return False
            # Filter all-uppercase OCR garbage chunks.
            if sum(1 for tok in alpha_toks if tok.isupper() and len(tok) > 2) >= 2:
                return False
            return True

        try:
            ents = shared_perform_ner(t)
            names = sanitize_names(ents.get("names") or [])
        except Exception:
            names = []
        role_hits = []
        role_patterns = [
            r"\b(?:name|holder name|owner name|applicant name)\s*[:\-]\s*([A-Za-z][A-Za-z\s\.\-']{2,80})",
            r"\b(?:vendor|purchaser|lessor|lessee|tenant|landlord|donor|donee|petitioner|respondent)\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.\-']{2,80})",
            r"\b(?:signed by|signature of|authorized signatory)\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.\-']{2,80})",
        ]
        for pat in role_patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                role_hits.append(m.group(1).strip())
        if not names:
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", t):
                role_hits.append(m.group(1).strip())
        merged = []
        for cand in list(names) + role_hits:
            cleaned = _clean_person_name(cand)
            if _is_likely_person_name(cleaned):
                merged.append(cleaned)
        out = []
        seen = set()
        for nm in merged:
            key = nm.lower()
            if key not in seen:
                seen.add(key)
                out.append(nm)
            if len(out) >= 8:
                break
        return ", ".join(out) if out else "No clear person names detected in the document."
    if any(k in q for k in ["email", "mail id", "e-mail"]):
        emails = list(dict.fromkeys(re.findall(EMAIL_PATTERN, t)))
        return ", ".join(emails[:5]) if emails else "No emails detected"
    if any(k in q for k in ["phone", "mobile", "contact number", "contact"]):
        phones = extract_phones_validated(t)
        out = [p if isinstance(p, str) else (p.get("normalized") or p.get("number") or "") for p in phones]
        out = [x for x in out if x]
        return ", ".join(out[:5]) if out else "No phone numbers detected"
    if "pan" in q:
        pans = re.findall(r"\b[A-Z]{5}\d{4}[A-Z]\b", t)
        pans = list(dict.fromkeys(pans))
        return ", ".join(pans[:3]) if pans else "PAN not found"
    if any(k in q for k in ["aadhaar", "aadhar", "uidai"]):
        nums = [m.group(0) for m in re.finditer(r"\b\d{12}\b", t)]
        return ", ".join(nums[:3]) if nums else "Aadhaar not found"
    if any(k in q for k in ["date", "dated", "timeline"]):
        dates = []
        for m in re.finditer(DATE_PATTERN, t):
            raw = m.group(0)
            try:
                parsed = dateparser.parse(raw)
                dates.append(parsed.strftime("%d/%m/%Y") if parsed else raw)
            except Exception:
                dates.append(raw)
        dates = list(dict.fromkeys(dates))
        return ", ".join(dates[:8]) if dates else "No dates detected"
    # Fallback: sentence overlap
    q_tokens = set(re.findall(r"\w+", q)) - {"what", "is", "the", "of", "in", "a", "an", "to"}
    if not q_tokens:
        return "Could not understand question."
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", t)
    best_sent = "Not found."
    max_overlap = 0
    for s in sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        overlap = len(q_tokens.intersection(s_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sent = s.strip()
    if max_overlap > 0:
        return best_sent
    return "Not available in the document"

def extract_key_facts_regex(full_text: str) -> Dict[str, str]:
    text = full_text.lower()
    
    # 1. Term/Duration
    term_match = re.search(r"(period|term|duration)\s*of\s*(this\s*agreement|lease)?\s*[:\-]?\s*(\d+\s*(months?|years?)|[\d\.]+\s*(months?|years?))", text)
    term_text = term_match.group(0) if term_match else "Term: Not explicitly found."
    
    # 2. Rent/Fees
    rent_match = re.search(r"(rent|monthly\s*rent|consideration|fees)\s*[:\-]?\s*(rs\.?|inr)?\s*([\d,]+)", text)
    rent_text = f"Rent/Fee detected: {rent_match.group(0)}" if rent_match else "Rent or payment not clearly detected."
    
    # 3. Access/Inspection
    access_text = "No clear access clause."
    if "access" in text or "inspection" in text or "entry" in text:
        access_text = "Clause regarding access/inspection is present."

    return {"term_text": term_text, "rent_text": rent_text, "access_text": access_text}

def extract_key_facts_llm(full_text: str) -> Dict[str, str]:
    return extract_key_facts_regex(full_text)

async def extract_key_facts_llm_async(full_text: str) -> Dict[str, str]:
    return await asyncio.to_thread(extract_key_facts_llm, full_text)

_humanizer_pipeline = None

def _get_humanizer():
    global _humanizer_pipeline
    if _humanizer_pipeline is None:
        try:
            from transformers import pipeline
            # Use a small model to keep it fast/local-friendly. t5-small is good for simple "translate to simple English" tasks.
            # or simplif/text-simplification-t5-base if available, but let's stick to summarization for safety if generic models aren't present.
            # better: "philschmid/bart-large-cnn-samsum" is good for conversational summaries, but heavy.
            # Let's use a standard summarizer as a proxy for "humanizing/simplifying"
            _humanizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            _humanizer_pipeline = False
    return _humanizer_pipeline

def humanize_text_ai(text: str) -> str:
    """
    Simplifies complex legal text using a local Transformer model (DistilBART),
    restoring the 'Humanize' feature without OpenAI.
    """
    try:
        pipe = _get_humanizer()
        if pipe:
            # Chunking effectively
            input_text = text[:1024] 
            summary = pipe(input_text, max_length=150, min_length=40, do_sample=False)
            return summary[0]['summary_text']
    except Exception:
        pass
        
    return summarize_fallback(text)

def legal_explanation_engine(text: str) -> str:
    return humanize_text_ai(text)

async def humanize_text_ai_async(text: str) -> str:
    return await asyncio.to_thread(humanize_text_ai, text)

async def analyze_text_overall_async(full_text: str):
    full_text = (full_text or "")
    if not full_text.strip():
        return {}, {}, ""
    if len(full_text.strip()) < 10:
        return {}, {}, full_text

    # 1. Pre-processing: Detect Language and Correct/Translate (Sequential because correction depends on translation)
    lang = detect_language(full_text[:2000])
    working_text = ""
    
    if lang != "English":
        print(f"[DEBUG] analysis_service: Non-English text detected ({lang}), translating...")
        working_text = await translate_to_english_async(full_text[:6000], lang)
    else:
        print("[DEBUG] analysis_service: English text detected, correcting OCR errors...")
        working_text = await correct_ocr_errors_async(full_text[:6000])
        
    if not working_text.strip():
        working_text = full_text[:6000]

    # 2. Extract Entities and Other Analysis (Parallel)
    # NER depends on working_text, but can run in parallel with regex extractions and spaCy summarization
    
    async def get_ner():
        return await perform_ner_async(working_text)

    def _regex_extractions(text, ftext):
        emails = re.findall(EMAIL_PATTERN, ftext)
        phones = extract_phones_validated(ftext)
        phone_details = []
        for s in phones:
            digits = re.sub(r"\D", "", s or "")
            normalized = f"+91 {digits}" if len(digits) == 10 and digits[:1] in "6789" else s
            tag = phone_compliance_tag(s)
            phone_details.append({"number": normalized, "normalized": normalized, "tag": tag})
        
        monetary_values = [f"Rs {m.group(2)}" for m in re.finditer(r"(rs\.?|inr)\s*([0-9][0-9,\.]*)", ftext, flags=re.IGNORECASE)]
        survey_numbers = list(set(m.group(2) for m in re.finditer(r"(survey\s*no\.?|s\.?no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", ftext, flags=re.IGNORECASE)))
        case_numbers = list(set(m.group(3) for m in re.finditer(r"((case|c\\.?s\\.?|o\\.?s\\.?|s\\.?a\\.?|w\\.?p\\.?)\\s*no\\.?)[\\s:\\-]*([A-Za-z0-9/\\-]+)", ftext, flags=re.IGNORECASE)))
        authorities = list(set(m.group(1).title() for m in re.finditer(r"(supreme court|high court|district court|consumer forum|tribunal|sub-registrar|registrar|rera)", ftext, flags=re.IGNORECASE)))
        dates = [str(dateparser.parse(d).date()) for d in re.findall(DATE_PATTERN, ftext) if dateparser.parse(d)]
        return {
            "emails": emails, "phone_details": phone_details, "monetary_values": monetary_values,
            "survey_numbers": survey_numbers, "case_numbers": case_numbers, 
            "authorities": authorities, "dates": dates
        }

    # Wrap sync spaCy ops in a thread if needed, but for now just run sequentially with NER
    ner_task = asyncio.create_task(get_ner())
    regex_task = asyncio.create_task(asyncio.to_thread(_regex_extractions, working_text, full_text))
    
    # Wait for NER and Regex
    ner_results, reg_results = await asyncio.gather(ner_task, regex_task)
    
    names = sanitize_names(ner_results.get("names", []))
    orgs = sanitize_orgs(ner_results.get("organizations", []))
    signers = extract_signatures(full_text)
    try:
        indian = extract_indian_entities(working_text or full_text)
    except Exception:
        indian = {"parties": [], "role_parties": [], "registration_numbers": [], "survey_numbers": [], "case_numbers": [], "courts": [], "land_types": [], "dates": [], "money": [], "property_details": "", "role_parties": []}
    if indian:
        names = sanitize_names((names or []) + (indian.get("parties") or []) + (indian.get("role_parties") or []) + (signers or []))
    
    emails = reg_results["emails"]
    phone_details = reg_results["phone_details"]
    monetary_values = reg_results["monetary_values"]
    survey_numbers = reg_results["survey_numbers"]
    case_numbers = reg_results["case_numbers"]
    authorities = reg_results["authorities"]
    dates = reg_results["dates"]
    try:
        low = (full_text or "").lower()
        tamil_present = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in (full_text or ""))
        property_cues = ("non judicial" in low) or ("stamp paper" in low) or ("sub registrar" in low) or ("stamp vendor" in low)
        if tamil_present and property_cues:
            tn = extract_tamil_names(full_text or "")
            tn_lat = [transliterate_tamil_to_latin(x) for x in tn if x]
            names = sanitize_names((names or []) + tn_lat)
        authority_whitelist = ["Sub Registrar", "Stamp Vendor", "Government of Tamil Nadu", "Revenue Department", "District Registrar"]
        detected_auths = []
        for aterm in authority_whitelist:
            if aterm.lower() in low:
                detected_auths.append(aterm)
        if detected_auths:
            orgs = sanitize_orgs((orgs or []) + detected_auths)
    except Exception:
        pass

    # SpaCy Analysis (Summary and Keywords)
    def _spacy_analysis(text):
        nlp = _get_nlp()
        if not nlp:
            return "No summary available.", {}
        
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        summary = "No summary available."
        keyword_frequency = {}
        
        if sentences:
            freq = Counter(t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct)
            maxf = max(freq.values()) if freq else 1
            
            sent_scores = {}
            for sent in doc.sents:
                s_text = sent.text.strip()
                if len(s_text) > 20:
                    score = sum(freq.get(t.lemma_.lower(), 0) / maxf for t in sent if not t.is_stop and not t.is_punct)
                    sent_scores[s_text] = score
            
            summary = " ".join(s for s, _ in sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:3])
        
        keyword_frequency = dict(Counter(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop).most_common(10))
        return summary, keyword_frequency

    summary, keyword_frequency = await asyncio.to_thread(_spacy_analysis, working_text)
    summary_source_text = working_text or full_text
    if _is_resume_like(summary_source_text) or _summary_needs_fix(summary):
        summary = summarize_fallback(summary_source_text)
    try:
        summary = _normalize_spaces(correct_ocr_errors(summary) or summary)
    except Exception:
        summary = _normalize_spaces(summary)
    if _summary_needs_fix(summary):
        summary = _build_structured_summary(
            doc_type="",
            names=names,
            orgs=orgs,
            dates=dates,
            clause_counter=Counter(),
            source_text=working_text or full_text,
        )
    land_classification = extract_land_classification(full_text)

    # Clause Detection
    search_text = working_text if len(working_text) > 100 else full_text
    clause_counter = Counter()
    for kw in CLAUSE_KEYWORDS:
        clause_counter[kw] = len(re.findall(r"\b" + re.escape(kw) + r"\b", search_text, flags=re.IGNORECASE))
    if _summary_needs_fix(summary) or _looks_like_ocr_noise(summary):
        summary = _build_structured_summary(
            doc_type="",
            names=names,
            orgs=orgs,
            dates=dates,
            clause_counter=clause_counter,
            source_text=working_text or full_text,
        )
    if _summary_needs_fix(summary):
        summary = "No clean summary available from OCR text. Please upload a clearer scan."
    
    # Aggregate
    results = {
        "summary": summary,
        "entities": {"names": names, "organizations": orgs},
        "signers": signers,
        "contact_info": {"emails": list(set(emails)), "phones": phone_details},
        "monetary_values": list(set(monetary_values)),
        "dates": list(set(dates)),
        "clauses_found": dict(clause_counter),
        "metadata": {
            "survey_numbers": list(set((survey_numbers or []) + (indian.get("survey_numbers") or []))),
            "case_numbers": list(set((case_numbers or []) + (indian.get("case_numbers") or []))),
            "authorities": list(set((authorities or []) + (indian.get("courts") or []))),
            "keyword_frequency": keyword_frequency,
            "land_classification": land_classification
        }
    }
    try:
        print(f"[ANALYSIS] text_len={len((working_text or full_text) or '')} names={len(names)} orgs={len(orgs)} emails={len(set(emails))} phones={len(phone_details)} dates={len(set(dates))} clauses_pos={sum(1 for _, c in clause_counter.items() if c>0)} summary_len={len(summary or '')}")
    except Exception:
        pass
    
    analytics = {
        "detected_language": lang,
        "word_count": len(full_text.split()),
        "summary": summary,
        "entities_count": len(names) + len(orgs),
        "risk_clauses_count": sum(1 for kw, count in clause_counter.items() if count > 0),
        "names": names,
        "organizations": orgs,
        "land_classification": land_classification
    }
    
    return results, analytics, working_text

def analyze_text_overall(full_text: str):
    # This sync wrapper is for backward compatibility if needed, 
    # but we should use the async version in pipeline.py
    import asyncio
    return asyncio.run(analyze_text_overall_async(full_text))

async def build_timeline_async(full_text: str) -> List[Dict[str, Any]]:
    """
    Module 8: Timeline Extraction.
    Uses TimelineService for robust event-date linking and chronological sorting.
    """
    try:
        res = await asyncio.to_thread(timeline_service.extract_timeline, full_text)
        events = res.get("timeline", [])
    except Exception as e:
        print(f"[DEBUG] build_timeline_async error: {e}")
        events = []
    
    if not events and is_openai_ready():
        try:
            prompt = (
                "Extract a chronological timeline of key events from the legal document. "
                "Include effective dates, execution dates, payment deadlines, and termination events. "
                "Return ONLY a JSON array of objects with keys \"date\" (YYYY-MM-DD) and \"event\". "
                "Document:\n" + full_text[:4000]
            )
            resp = await get_async_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600
            )
            raw = (resp.choices[0].message.content or "").strip()
            import json
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1:
                events = json.loads(raw[start:end])
        except Exception as e:
            print(f"[DEBUG] build_timeline_async OpenAI error: {e}")
            try:
                msg = str(e).lower()
                if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
                    mark_quota_exhausted()
            except Exception:
                pass
            
    return sorted(events, key=lambda x: x.get("date", ""))[:20]

async def validate_signers_async(full_text: str, signers: List[str]) -> Dict[str, Any]:
    if not is_openai_ready():
        return {"signers": [], "overall_signer_score": None}
    try:
        prompt = (
            "Validate signers found in the document. Return strict JSON with keys: "
            "\"signers\": [{\"name\":\"...\",\"present\":true/false,\"role\":\"...\"," 
            "\"issues\":[\"...\"],\"score\":0-100}],\"overall_signer_score\":0-100. "
            "Consider whether signatures, dates, roles, and authorization are present. "
            "Text:\n" + full_text[:6000] + "\n\nSigners:\n" + json.dumps(signers[:10])
        )
        resp = await get_async_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        print(f"[DEBUG] validate_signers_async error: {e}")
        try:
            msg = str(e).lower()
            if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
                mark_quota_exhausted()
        except Exception:
            pass
        return {"signers": [], "overall_signer_score": None}

async def jurisdiction_specific_checks_async(full_text: str) -> Dict[str, Any]:
    # Try OpenAI first
    if is_openai_ready():
        try:
            prompt = (
                "Infer jurisdiction from the document and perform jurisdiction-specific checks. "
                "Focus on Indian Contract Act if applicable; otherwise provide general contract checks. "
                "Return strict JSON with keys: "
                "\"jurisdiction\":\"country or region\","
                "\"checks\":[{\"rule\":\"...\",\"status\":\"pass/fail\",\"detail\":\"...\",\"severity\":\"Low/Medium/High\"}],"
                "\"overall_compliance_score\":0-100. Text:\n" + full_text[:6000]
            )
            resp = await get_async_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            print(f"[DEBUG] jurisdiction_specific_checks_async OpenAI error: {e}")
            try:
                msg = str(e).lower()
                if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
                    mark_quota_exhausted()
            except Exception:
                pass
            
    # Fallback to Rule-based in compliance_service
    try:
        from backend.app.services.compliance_service import jurisdiction_checks
        return jurisdiction_checks(full_text)
    except:
        return {"jurisdiction": None, "checks": [], "overall_compliance_score": 0}

async def curated_jurisdiction_templates_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(curated_jurisdiction_templates, full_text)

def classify_document_type(full_text: str) -> str:
    t = (full_text or "").strip()
    if not t:
        return "Generic Legal Document"

    try:
        rule_label = classifier.detect_document_type_by_rules(full_text)
        if rule_label:
            return rule_label

        res = classifier.predict_document_type(full_text)
        prediction, confidence = res[0], res[1]

        if confidence < 0.50 or prediction in ["Generic Legal Document", "Unknown", "Other Document"]:
            if rule_label:
                return rule_label

        return prediction or "Generic Legal Document"
    except Exception as e:
        print(f"[DEBUG] classify_document_type error: {e}")
        return "Generic Legal Document"

async def classify_document_type_async(full_text: str) -> str:
    return await asyncio.to_thread(classify_document_type, full_text)

def is_legal_document(full_text: str, doc_type: str = None) -> bool:
    t = (full_text or "").lower()
    legal_keywords = [
        "agreement","deed","affidavit","stamp duty","notary","parties","witness",
        "jurisdiction","court","petitioner","respondent","schedule","registrar",
        "registration","sub-registrar","lease","contract"
    ]
    if doc_type is None:
        doc_type = classify_document_type(full_text)
    if doc_type in {
        "Sale Deed","Settlement Deed","Partition Deed","Release Deed","Rectification Deed",
        "Rental Agreement","Lease Deed","Power of Attorney","Affidavit","Legal Agreement","Contract",
        "Court Order","Court Judgment","Legal Notice","Petition","Encumbrance Certificate","Layout Approval",
        "Gift Deed","Partnership Deed","Employment Contract","Corporate Agreement"
    }:
        return True
    if doc_type in {"Resume","Invoice","ID Card"}:
        return False
    score = sum(1 for k in legal_keywords if k in t)
    return score >= 2

def is_indian_legal(full_text: str) -> bool:
    t = (full_text or "").lower()
    ind_indicators = [
        "india","indian","rupees","rs.","sub-registrar","stamp duty","schedule property",
        "cmda","dtcp","patta","chitta","encumbrance","fmb","tslr","high court","supreme court","district court"
    ]
    j = jurisdiction_checks(full_text) or {}
    if (j.get("jurisdiction") or "").lower() == "india":
        return True
    doc_type = classify_document_type(full_text)
    indian_types = {
        "Sale Deed","Lease Deed","Rental Agreement","Gift Deed","Patta","TSLR",
        "Encumbrance Certificate","Court Judgment","Court Order","Legal Notice","Affidavit",
        "Employment Contract","Partnership Deed","Corporate Agreement","Settlement Deed",
        "Partition Deed","Release Deed","Rectification Deed"
    }
    if doc_type in indian_types:
        return True
    return any(k in t for k in ind_indicators)

def classify_legal_two_step(full_text: str, doc_type: Optional[str] = None) -> dict:
    text = full_text or ""
    doc_type = doc_type or classify_document_type(text)
    legal = is_legal_document(text, doc_type)
    indian = False
    if legal:
        indian = is_indian_legal(text)
    return {
        "is_legal": legal,
        "is_indian": indian,
        "document_type": doc_type
    }

async def ai_classify_and_extract_async(full_text: str) -> Dict[str, Any]:
    t = (full_text or "")[:6000]
    if not is_openai_ready():
        return {
            "category": "Other",
            "document_type": "Other",
            "entities": {
                "parties": [],
                "dates": [],
                "money": [],
                "property_details": "",
                "survey_numbers": [],
                "registration_numbers": [],
                "case_numbers": [],
                "courts": [],
                "land_types": []
            }
        }
    prompt = (
        "Assume Indian context only. Use Indian legal terminology.\n"
        "Task: Classify the document and extract entities.\n"
        "Return STRICT JSON with keys:\n"
        "{\n"
        "  \"category\": \"Legal Document|Land/Property Document|Identity Document|Certificate|Financial Document|Employment Document|Other\",\n"
        "  \"document_type\": \"Exact type from supported list or Other\",\n"
        "  \"entities\": {\n"
        "    \"parties\": [],\n"
        "    \"dates\": [],\n"
        "    \"money\": [],\n"
        "    \"property_details\": \"\",\n"
        "    \"survey_numbers\": [],\n"
        "    \"registration_numbers\": [],\n"
        "    \"case_numbers\": [],\n"
        "    \"courts\": [],\n"
        "    \"land_types\": []\n"
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Dates format: DD/MM/YYYY\n"
        "- Currency: ₹ format\n"
        "- If data missing, return empty lists\n"
        "- Always return valid JSON only, no text outside JSON\n\n"
        "Text:\n" + t
    )
    try:
        resp = await get_async_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        ents = data.get("entities") or {}
        # Normalize dates to DD/MM/YYYY
        normalized_dates = []
        for d in ents.get("dates") or []:
            try:
                parsed = dateparser.parse(d)
                if parsed:
                    normalized_dates.append(parsed.strftime("%d/%m/%Y"))
                else:
                    normalized_dates.append(d)
            except Exception:
                normalized_dates.append(d)
        ents["dates"] = normalized_dates
        data["entities"] = ents
        return data
    except Exception as e:
        try:
            msg = str(e).lower()
            if "insufficient_quota" in msg or "quota" in msg or "billing" in msg:
                mark_quota_exhausted()
        except Exception:
            pass
        return {
            "category": "Other",
            "document_type": "Other",
            "entities": {
                "parties": [],
                "dates": [],
                "money": [],
                "property_details": "",
                "survey_numbers": [],
                "registration_numbers": [],
                "case_numbers": [],
                "courts": [],
                "land_types": []
            },
            "error": str(e)
        }
def classify_document_category(full_text: str) -> str:
    t = (full_text or "")
    doc_type = classify_document_type(t)
    land_types = {
        "Patta","Chitta","TSLR","FMB Sketch","Layout Approval","Sale Deed","Settlement Deed","Land Property",
        "Partition Deed","Release Deed","Rectification Deed","Encumbrance Certificate","Mortgage Deed",
        "Power of Attorney","Gift Deed"
    }
    legal_types = {
        "Court Judgment","Court Order","Legal Notice","Affidavit","Petition",
        "Corporate Agreement","Partnership Deed","MOA / AOA","Regulatory Filing","Legal Agreement","Contract"
    }
    identity_types = {
        "Aadhaar Card","e-Aadhaar","Masked Aadhaar","Aadhaar PVC Card","Indian Passport","Passport","Visa",
        "OCI Card","PIO Card","Voter ID","e-Voter ID","Driving Licence","Learner Licence","Learner Licence","e-Driving Licence"
    }
    certificate_types = {
        "Birth Certificate","Death Certificate","Marriage Certificate","Caste Certificate","Income Certificate",
        "Domicile Certificate","Disability Certificate","Residence Certificate","Bonafide Certificate",
        "Migration Certificate","Legal Heir Certificate","PUC Certificate","Transfer Certificate"
    }
    financial_types = {
        "PAN Card","e-PAN","GST Certificate","Bank Passbook","Bank Statement","Cancelled Cheque","Credit Card Statement",
        "Motor Insurance","Vehicle RC"
    }
    employment_types = {
        "Employment Contract","Employee ID","Experience Certificate","Salary Certificate"
    }
    if doc_type in land_types:
        return "Land/Property Document"
    if doc_type in legal_types:
        return "Legal Document"
    if doc_type in identity_types:
        return "Identity Document"
    if doc_type in certificate_types:
        return "Certificate"
    if doc_type in financial_types:
        return "Financial Document"
    if doc_type in employment_types:
        return "Employment Document"
    if doc_type in {"Resume","Invoice","ID Card","Application Interface Screenshot"}:
        return "Non-Legal Document"
    return "Other"

async def classify_document_category_async(full_text: str) -> str:
    return await asyncio.to_thread(classify_document_category, full_text)


def deterministic_classify_document(full_text: str) -> Dict[str, str]:
    text = (full_text or "").strip()
    lower = text.lower()
    first500 = text[:500]
    word_count = len(re.findall(r"\b\w+\b", text))
    stability_hash = hashlib.sha256(f"{first500}|{word_count}".encode("utf-8", errors="ignore")).hexdigest()

    if not text:
        return {
            "document_type": "UNKNOWN",
            "rule_match_score": "0%",
            "ml_confidence": "0%",
            "final_confidence": "0%",
            "classification_source": "ML",
            "stability_hash": stability_hash,
        }

    rule_sets: Dict[str, List[str]] = {
        "Patta": [
            "patta number", "survey number", "taluk", "village", "revenue department", "district", "tslr",
        ],
        "Sale Deed": [
            "vendor", "purchaser", "sale consideration", "schedule property", "registered at sub registrar",
        ],
        "Lease Deed": [
            "lessor", "lessee", "lease period", "monthly rent", "security deposit",
        ],
        "Court Order": [
            "in the court of", "petitioner", "respondent", "case no", "order dated",
        ],
        "Non Judicial Stamp Paper": [
            "non judicial", "india non judicial", "stamp paper", "stamp vendor", "lic no", "tamil nadu",
        ],
    }

    def _contains_phrase(phrase: str) -> bool:
        return phrase in lower

    # STEP 1: rule-priority strict matching.
    best_label = ""
    best_hits = 0
    best_total = 1
    for label, phrases in rule_sets.items():
        hits = sum(1 for p in phrases if _contains_phrase(p))
        if hits > best_hits:
            best_hits = hits
            best_total = len(phrases)
            best_label = label

    rule_pct = int(round((best_hits / max(1, best_total)) * 100))

    # STEP 2: structural validation.
    has_title_header = any(k in lower for k in ["deed", "order", "certificate", "patta", "court"])
    has_authority_ref = any(k in lower for k in ["sub registrar", "revenue department", "court", "tahsildar", "district"])
    has_format_pattern = any(k in lower for k in ["no.", "number", "dated", "date", "case no", "survey number", ":"])
    structure_score = int(has_title_header) + int(has_authority_ref) + int(has_format_pattern)

    if best_label == "Non Judicial Stamp Paper" and best_hits >= 2:
        return {
            "document_type": "Non Judicial Stamp Paper",
            "rule_match_score": f"{rule_pct}%",
            "ml_confidence": "0%",
            "final_confidence": "92%",
            "classification_source": "RULE",
            "stability_hash": stability_hash,
        }

    if best_hits >= 3:
        # Deterministic high-confidence rule result.
        base_conf = 96
        if structure_score == 2:
            base_conf = 94
        elif structure_score <= 1:
            base_conf = 90
        return {
            "document_type": best_label,
            "rule_match_score": f"{rule_pct}%",
            "ml_confidence": "0%",
            "final_confidence": f"{base_conf}%",
            "classification_source": "RULE",
            "stability_hash": stability_hash,
        }

    # STEP 3: ML fallback only when no rule threshold match.
    try:
        res = classifier.predict_document_type(text)
        pred, conf, m_version = res[0], res[1], res[2]
        ml_pct = int(round(max(0.0, min(1.0, float(conf or 0.0))) * 100))
        if ml_pct < 60:
            return {
                "document_type": "UNKNOWN",
                "rule_match_score": f"{rule_pct}%",
                "ml_confidence": f"{ml_pct}%",
                "final_confidence": f"{ml_pct}%",
                "classification_source": "ML",
                "stability_hash": stability_hash,
                "model_version": m_version,
            }
        return {
            "document_type": str(pred or "UNKNOWN"),
            "rule_match_score": f"{rule_pct}%",
            "ml_confidence": f"{ml_pct}%",
            "final_confidence": f"{ml_pct}%",
            "classification_source": "ML",
            "stability_hash": stability_hash,
            "model_version": m_version,
        }
    except Exception:
        return {
            "document_type": "UNKNOWN",
            "rule_match_score": f"{rule_pct}%",
            "ml_confidence": "0%",
            "final_confidence": "0%",
            "classification_source": "ML",
            "stability_hash": stability_hash,
        }


def structural_audit_strict(full_text: str, classified_type: Optional[str] = None) -> Dict[str, Any]:
    text = (full_text or "").strip()
    lower = text.lower()
    if not text:
        return {
            "document_type": "NOT A LEGAL DOCUMENT",
            "sections_missing": [],
            "note": "No legal structural template applicable."
        }

    non_legal_types = {
        "Resume", "Invoice", "ID Card", "Driving Licence", "Bank Statement", "Marksheet",
        "Admit Card", "Transfer Certificate", "Bonafide Certificate", "Other Non Legal Document",
        "Application Interface Screenshot"
    }

    detected_raw = (classified_type or classify_document_type(text) or "").strip()
    detected_norm = classifier.canonicalize_category(detected_raw) if detected_raw else "Other"
    if detected_norm in non_legal_types:
        return {
            "document_type": "NOT A LEGAL DOCUMENT",
            "sections_missing": [],
            "note": "No legal structural template applicable."
        }

    allowed_types = {
        "Sale Deed",
        "Lease Deed",
        "Rental Agreement",
        "Gift Deed",
        "Patta",
        "Encumbrance Certificate",
        "Court Judgment",
        "Court Order",
        "Legal Notice",
    }
    doc_type = detected_norm if detected_norm in allowed_types else "Other"

    templates: Dict[str, List[Dict[str, Any]]] = {
        "Sale Deed": [
            {"section_name": "Parties (Vendor & Purchaser)", "groups": [["vendor", "seller"], ["purchaser", "buyer"]], "severity": "Critical"},
            {"section_name": "Property Description (Schedule)", "groups": [["schedule property", "property schedule", "description of property"]], "severity": "Critical"},
            {"section_name": "Consideration Amount", "groups": [["consideration", "sale consideration", "consideration amount"]], "severity": "Critical"},
            {"section_name": "Payment Terms", "groups": [["payment", "paid", "mode of payment", "payment terms"]], "severity": "Moderate"},
            {"section_name": "Registration Details", "groups": [["registered", "registration number", "sub registrar", "doc no"]], "severity": "Critical"},
            {"section_name": "Witness Section", "groups": [["witness", "witnesses"]], "severity": "Moderate"},
            {"section_name": "Signature Block", "groups": [["signature", "signed by", "executant"]], "severity": "Critical"},
        ],
        "Lease Deed": [
            {"section_name": "Lessor", "groups": [["lessor", "landlord"]], "severity": "Critical"},
            {"section_name": "Lessee", "groups": [["lessee", "tenant"]], "severity": "Critical"},
            {"section_name": "Lease Period", "groups": [["lease period", "term", "duration"]], "severity": "Critical"},
            {"section_name": "Rent Amount", "groups": [["rent", "monthly rent"]], "severity": "Critical"},
            {"section_name": "Security Deposit", "groups": [["security deposit", "advance"]], "severity": "Moderate"},
            {"section_name": "Termination Clause", "groups": [["termination", "terminate", "notice period"]], "severity": "Moderate"},
            {"section_name": "Dispute Clause", "groups": [["dispute", "arbitration", "jurisdiction"]], "severity": "Moderate"},
            {"section_name": "Signatures", "groups": [["signature", "signed by"]], "severity": "Critical"},
        ],
        "Rental Agreement": [
            {"section_name": "Landlord", "groups": [["landlord", "owner"]], "severity": "Critical"},
            {"section_name": "Tenant", "groups": [["tenant"]], "severity": "Critical"},
            {"section_name": "Rental Term", "groups": [["term", "duration", "period"]], "severity": "Critical"},
            {"section_name": "Rent Amount", "groups": [["rent", "monthly rent"]], "severity": "Critical"},
            {"section_name": "Deposit/Advance", "groups": [["security deposit", "advance"]], "severity": "Moderate"},
            {"section_name": "Termination Clause", "groups": [["termination", "notice period"]], "severity": "Moderate"},
            {"section_name": "Signatures", "groups": [["signature", "signed by"]], "severity": "Critical"},
        ],
        "Gift Deed": [
            {"section_name": "Gift Declaration", "groups": [["gift deed", "gift"]], "severity": "Critical"},
            {"section_name": "Donor", "groups": [["donor"]], "severity": "Critical"},
            {"section_name": "Donee", "groups": [["donee"]], "severity": "Critical"},
            {"section_name": "Property Description", "groups": [["schedule property", "property schedule", "property"]], "severity": "Critical"},
            {"section_name": "Registration Details", "groups": [["registered", "sub registrar", "registration number"]], "severity": "Moderate"},
            {"section_name": "Signature Block", "groups": [["signature", "signed by"]], "severity": "Critical"},
        ],
        "Patta": [
            {"section_name": "Patta Number", "groups": [["patta number"]], "severity": "Critical"},
            {"section_name": "Survey Number", "groups": [["survey number", "s.no", "tslr"]], "severity": "Critical"},
            {"section_name": "Village", "groups": [["village"]], "severity": "Critical"},
            {"section_name": "Taluk", "groups": [["taluk"]], "severity": "Critical"},
            {"section_name": "District", "groups": [["district"]], "severity": "Moderate"},
            {"section_name": "Land Owner Name", "groups": [["owner name", "name of pattadar", "pattadar", "holder name"]], "severity": "Critical"},
            {"section_name": "Revenue Authority Reference", "groups": [["revenue department", "tahsildar", "authority"]], "severity": "Moderate"},
        ],
        "Encumbrance Certificate": [
            {"section_name": "EC Reference Number", "groups": [["ec number", "encumbrance certificate"]], "severity": "Critical"},
            {"section_name": "Sub-Registrar Details", "groups": [["sub registrar", "sub registrar office", "sro"]], "severity": "Critical"},
            {"section_name": "Property/Survey Reference", "groups": [["survey number", "property"]], "severity": "Moderate"},
            {"section_name": "Transaction/Entry Details", "groups": [["encumbrance", "entry", "document number", "registration"]], "severity": "Moderate"},
        ],
        "Court Judgment": [
            {"section_name": "Court Name", "groups": [["in the court of", "high court", "district court", "supreme court"]], "severity": "Critical"},
            {"section_name": "Case Number", "groups": [["case no", "case number", "w.p.", "c.r.p."]], "severity": "Critical"},
            {"section_name": "Petitioner", "groups": [["petitioner"]], "severity": "Critical"},
            {"section_name": "Respondent", "groups": [["respondent"]], "severity": "Critical"},
            {"section_name": "Date", "groups": [["dated", "pronounced on", "date"]], "severity": "Moderate"},
            {"section_name": "Judge Name", "groups": [["judge", "justice", "hon'ble"]], "severity": "Moderate"},
            {"section_name": "Final Judgment Section", "groups": [["judgment", "ordered", "disposed of"]], "severity": "Critical"},
        ],
        "Court Order": [
            {"section_name": "Court Name", "groups": [["in the court of", "high court", "district court", "supreme court"]], "severity": "Critical"},
            {"section_name": "Case Number", "groups": [["case no", "case number", "w.p.", "c.r.p."]], "severity": "Critical"},
            {"section_name": "Petitioner", "groups": [["petitioner"]], "severity": "Critical"},
            {"section_name": "Respondent", "groups": [["respondent"]], "severity": "Critical"},
            {"section_name": "Date", "groups": [["dated", "date"]], "severity": "Moderate"},
            {"section_name": "Judge Name", "groups": [["judge", "justice", "hon'ble"]], "severity": "Moderate"},
            {"section_name": "Final Order Section", "groups": [["order", "it is ordered", "directed"]], "severity": "Critical"},
        ],
        "Legal Notice": [
            {"section_name": "Notice Header", "groups": [["legal notice", "notice"]], "severity": "Critical"},
            {"section_name": "Sender/Advocate Details", "groups": [["advocate", "counsel", "on behalf of"]], "severity": "Moderate"},
            {"section_name": "Recipient Details", "groups": [["to,", "addressee", "recipient"]], "severity": "Moderate"},
            {"section_name": "Cause of Action", "groups": [["cause of action", "breach", "facts"]], "severity": "Critical"},
            {"section_name": "Demand/Relief", "groups": [["called upon", "demand", "pay", "comply"]], "severity": "Critical"},
            {"section_name": "Reply Timeline", "groups": [["within", "days", "failing which"]], "severity": "Moderate"},
            {"section_name": "Signature", "groups": [["signature", "signed"]], "severity": "Critical"},
        ],
        "Other": [
            {"section_name": "Document Header", "groups": [["deed", "agreement", "certificate", "order", "notice"]], "severity": "Moderate"},
            {"section_name": "Party/Entity Identification", "groups": [["name", "party", "petitioner", "respondent", "vendor", "purchaser"]], "severity": "Moderate"},
            {"section_name": "Date/Reference Details", "groups": [["date", "reference", "registration", "document number"]], "severity": "Moderate"},
            {"section_name": "Signature/Authority Block", "groups": [["signature", "signed", "authority", "seal"]], "severity": "Critical"},
        ],
    }

    required_sections = templates.get(doc_type, templates["Other"])

    def _status_for_section(section: Dict[str, Any]) -> Tuple[str, int, int]:
        groups = section.get("groups") or []
        total_groups = len(groups)
        matched_groups = 0
        for group in groups:
            if any((kw or "").lower() in lower for kw in (group or [])):
                matched_groups += 1
        if matched_groups == total_groups and total_groups > 0:
            return "PRESENT", matched_groups, total_groups
        if matched_groups > 0:
            return "PARTIAL", matched_groups, total_groups
        return "MISSING", matched_groups, total_groups

    sections_missing: List[Dict[str, str]] = []
    found_count = 0
    for sec in required_sections:
        status, matched, total = _status_for_section(sec)
        if status == "PRESENT":
            found_count += 1
        elif status == "PARTIAL":
            sections_missing.append({
                "section_name": sec.get("section_name", ""),
                "severity": sec.get("severity", "Moderate"),
                "reason": f"Partial match ({matched}/{total}) in document text."
            })
        else:
            sections_missing.append({
                "section_name": sec.get("section_name", ""),
                "severity": sec.get("severity", "Moderate"),
                "reason": "No reliable keywords for this section found in document text."
            })

    return {
        "document_type": doc_type,
        "total_required_sections": str(len(required_sections)),
        "sections_found": str(found_count),
        "sections_missing": sections_missing,
    }

def hybrid_classify_verify(full_text: str, filename: Optional[str] = None) -> Dict[str, Any]:
    text = (full_text or "").strip()
    lower = text.lower()
    if not text:
        return {
            "document_name": filename or "",
            "document_type": "UNKNOWN",
            "legal_status": "UNKNOWN",
            "confidence_score": "0%",
            "risk_level": "LOW",
            "missing_clauses": [],
            "reasoning_summary": "No text provided."
        }

    strong_rules = {
        "Sale Deed": [
            "consideration amount", "vendor", "purchaser", "schedule property", "sale deed", "registered at sub registrar"
        ],
        "Settlement Deed": [
            "settlement deed", "settlor", "settlee", "schedule property", "registered at sub registrar"
        ],
        "Partition Deed": [
            "partition deed", "coparceners", "share", "schedule property", "registered at sub registrar"
        ],
        "Release Deed": [
            "release deed", "releasor", "releasee", "relinquish", "schedule property"
        ],
        "Rectification Deed": [
            "rectification deed", "correction", "error in previous deed", "registered at sub registrar"
        ],
        "Mortgage Deed": [
            "mortgage deed", "mortgagor", "mortgagee", "secured loan", "redemption"
        ],
        "Power of Attorney": [
            "power of attorney", "principal", "attorney", "authorized", "to act on behalf"
        ],
        "Patta": [
            "patta number", "taluk", "village", "survey number", "revenue department", "tslr"
        ],
        "Lease Deed": [
            "lessor", "lessee", "lease period", "monthly rent", "security deposit"
        ],
        "Rental Agreement": [
            "rent agreement", "tenant", "landlord", "advance amount"
        ],
        "Gift Deed": [
            "gift deed", "donor", "donee", "out of love and affection"
        ],
        "Court Judgment": [
            "in the court of", "petitioner", "respondent", "order", "judgment"
        ],
        "Encumbrance Certificate": [
            "encumbrance", "sub registrar office", "ec number"
        ],
    }

    required_clause_rules = {
        "Sale Deed": [
            {"name": "Consideration Clause", "keywords": ["consideration amount", "sale consideration", "consideration"], "critical": True},
            {"name": "Parties Clause", "keywords": ["vendor", "purchaser"], "critical": True},
            {"name": "Schedule Property Clause", "keywords": ["schedule property", "property schedule"], "critical": True},
            {"name": "Registration Clause", "keywords": ["sub registrar", "registered as", "registration number"], "critical": True},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Settlement Deed": [
            {"name": "Settlement Declaration Clause", "keywords": ["settlement deed", "settlement"], "critical": True},
            {"name": "Parties Clause", "keywords": ["settlor", "settlee"], "critical": True},
            {"name": "Schedule Property Clause", "keywords": ["schedule property", "property schedule"], "critical": True},
            {"name": "Registration Clause", "keywords": ["sub registrar", "registered"], "critical": True},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Partition Deed": [
            {"name": "Partition Declaration Clause", "keywords": ["partition deed", "partition"], "critical": True},
            {"name": "Parties Clause", "keywords": ["coparceners", "co-parceners", "parties"], "critical": True},
            {"name": "Share Allocation Clause", "keywords": ["share", "allotment"], "critical": True},
            {"name": "Schedule Property Clause", "keywords": ["schedule property", "property schedule"], "critical": True},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Release Deed": [
            {"name": "Release Declaration Clause", "keywords": ["release deed", "relinquish"], "critical": True},
            {"name": "Parties Clause", "keywords": ["releasor", "releasee"], "critical": True},
            {"name": "Property/Right Clause", "keywords": ["schedule property", "rights", "interest"], "critical": True},
            {"name": "Registration Clause", "keywords": ["sub registrar", "registered"], "critical": False},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Rectification Deed": [
            {"name": "Rectification Declaration Clause", "keywords": ["rectification deed", "correction deed"], "critical": True},
            {"name": "Error Reference Clause", "keywords": ["error in previous deed", "typographical error", "mistake"], "critical": True},
            {"name": "Original Deed Reference Clause", "keywords": ["document no", "registration number", "original deed"], "critical": True},
            {"name": "Registration Clause", "keywords": ["sub registrar", "registered"], "critical": False},
        ],
        "Mortgage Deed": [
            {"name": "Mortgage Declaration Clause", "keywords": ["mortgage deed", "mortgage"], "critical": True},
            {"name": "Parties Clause", "keywords": ["mortgagor", "mortgagee"], "critical": True},
            {"name": "Secured Obligation Clause", "keywords": ["secured loan", "loan amount", "debt"], "critical": True},
            {"name": "Property Security Clause", "keywords": ["schedule property", "security for repayment"], "critical": True},
            {"name": "Redemption Clause", "keywords": ["redemption", "redeem"], "critical": False},
        ],
        "Power of Attorney": [
            {"name": "Authorization Clause", "keywords": ["power of attorney", "authorized", "to act on behalf"], "critical": True},
            {"name": "Parties Clause", "keywords": ["principal", "attorney"], "critical": True},
            {"name": "Scope of Powers Clause", "keywords": ["powers", "authority", "execute", "represent"], "critical": True},
            {"name": "Duration/Revocation Clause", "keywords": ["revocation", "valid until", "irrevocable"], "critical": False},
            {"name": "Signature/Attestation Clause", "keywords": ["signature", "signed", "witness", "notary"], "critical": False},
        ],
        "Lease Deed": [
            {"name": "Parties Clause", "keywords": ["lessor", "lessee"], "critical": True},
            {"name": "Lease Period Clause", "keywords": ["lease period", "term of lease"], "critical": True},
            {"name": "Rent Clause", "keywords": ["monthly rent", "rent"], "critical": True},
            {"name": "Security Deposit Clause", "keywords": ["security deposit", "advance"], "critical": False},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Rental Agreement": [
            {"name": "Parties Clause", "keywords": ["tenant", "landlord"], "critical": True},
            {"name": "Rent Clause", "keywords": ["rent agreement", "monthly rent", "rent"], "critical": True},
            {"name": "Advance Deposit Clause", "keywords": ["advance amount", "security deposit"], "critical": False},
            {"name": "Term Clause", "keywords": ["period", "term", "duration"], "critical": True},
            {"name": "Signature Clause", "keywords": ["signature", "signed", "witness"], "critical": False},
        ],
        "Gift Deed": [
            {"name": "Gift Declaration Clause", "keywords": ["gift deed", "out of love and affection"], "critical": True},
            {"name": "Parties Clause", "keywords": ["donor", "donee"], "critical": True},
            {"name": "Property Clause", "keywords": ["schedule property", "property"], "critical": True},
            {"name": "Registration Clause", "keywords": ["sub registrar", "registered"], "critical": False},
        ],
        "Patta": [
            {"name": "Patta Number Clause", "keywords": ["patta number"], "critical": True},
            {"name": "Location Clause", "keywords": ["taluk", "village"], "critical": True},
            {"name": "Survey Clause", "keywords": ["survey number", "tslr"], "critical": True},
            {"name": "Authority Clause", "keywords": ["revenue department", "tahsildar"], "critical": False},
        ],
        "Court Judgment": [
            {"name": "Court Header Clause", "keywords": ["in the court of"], "critical": True},
            {"name": "Party Clause", "keywords": ["petitioner", "respondent"], "critical": True},
            {"name": "Judgment Clause", "keywords": ["judgment", "order"], "critical": True},
            {"name": "Judge Signature Clause", "keywords": ["judge", "pronounced", "signature"], "critical": False},
        ],
        "Encumbrance Certificate": [
            {"name": "EC Identification Clause", "keywords": ["encumbrance", "ec number"], "critical": True},
            {"name": "Registrar Clause", "keywords": ["sub registrar office", "sub registrar"], "critical": True},
            {"name": "Property/Survey Clause", "keywords": ["survey number", "property"], "critical": False},
        ],
    }

    def _best_rule_match() -> Tuple[Optional[str], float]:
        best_label = None
        best_ratio = 0.0
        for label, keywords in strong_rules.items():
            hits = sum(1 for kw in keywords if kw in lower)
            ratio = hits / max(1, len(keywords))
            if ratio > best_ratio:
                best_label = label
                best_ratio = ratio
        return best_label, best_ratio

    rule_label, rule_ratio = _best_rule_match()
    ml_used = False
    ml_pred = None
    doc_type = "UNKNOWN"
    confidence_pct = int(round(max(0.0, min(1.0, rule_ratio)) * 100))

    # STEP 1: Rule-based validation. If strong match > 60%, skip ML.
    if rule_label and rule_ratio > 0.60:
        doc_type = rule_label
    else:
        # STEP 2: ML classification fallback.
        ml_used = True
        try:
            res = classifier.predict_document_type(text)
            ml_pred, ml_conf = res[0], res[1]
            ml_conf = float(ml_conf or 0.0)
            confidence_pct = int(round(max(0.0, min(1.0, ml_conf)) * 100))
            if ml_conf < 0.40:
                doc_type = "UNKNOWN"
            else:
                doc_type = (ml_pred or "UNKNOWN").strip() or "UNKNOWN"
        except Exception:
            doc_type = "UNKNOWN"
            confidence_pct = max(confidence_pct, 0)

    non_legal_markers = [
        "resume", "curriculum vitae", "invoice", "proforma invoice", "cover letter", "dear sir", "biodata"
    ]
    legal_markers = [
        "deed", "agreement", "court", "petitioner", "respondent", "patta", "encumbrance", "sub registrar", "judgment"
    ]
    non_legal_hits = sum(1 for m in non_legal_markers if m in lower)
    legal_hits = sum(1 for m in legal_markers if m in lower)
    non_legal_types = {
        "resume", "invoice", "id card", "driving licence", "bank statement", "other non legal document", "application interface screenshot"
    }
    if (non_legal_hits > 0 and legal_hits == 0) or ((doc_type or "").strip().lower() in non_legal_types and legal_hits == 0):
        return {
            "document_name": filename or "",
            "document_type": "NOT A LEGAL DOCUMENT",
            "legal_status": "NOT LEGAL",
            "confidence_score": "95%",
            "risk_level": "LOW",
            "missing_clauses": [],
            "reasoning_summary": "Document does not contain legal property or court-related structure."
        }

    # STEP 3: AI legal structure validation.
    structure_checks = {
        "official_structure": any(k in lower for k in [
            "whereas", "hereby", "in witness whereof", "this deed", "between", "thereof", "therein"
        ]),
        "proper_legal_language": any(k in lower for k in [
            "hereby", "whereas", "thereof", "witnesseth", "pursuant", "hereinafter"
        ]),
        "government_references": any(k in lower for k in [
            "government", "govt", "department", "sub registrar", "court", "tahsildar", "revenue department"
        ]),
        "signature_block": any(k in lower for k in [
            "signature", "signed by", "signed", "witness"
        ]),
        "registration_details": any(k in lower for k in [
            "registration number", "registered as", "doc no", "document no", "sub registrar", "ec number"
        ]),
        "stamp_duty_mention": any(k in lower for k in [
            "stamp duty", "e-stamp", "stamp paper", "non judicial stamp"
        ]),
        "seal_or_authority_mention": any(k in lower for k in [
            "seal", "authorized signatory", "authority", "notary", "registrar", "tahsildar"
        ]),
    }
    structure_score = sum(1 for v in structure_checks.values() if v)
    legal_structure_valid = (
        structure_score >= 4
        and structure_checks["proper_legal_language"]
        and (structure_checks["signature_block"] or structure_checks["registration_details"] or structure_checks["seal_or_authority_mention"])
    )
    no_legal_structure_detected = structure_score == 0
    partial_structure = 0 < structure_score < 4

    # Missing clauses by detected type (if applicable).
    missing_clauses: List[str] = []
    critical_missing = 0
    for rule in required_clause_rules.get(doc_type, []):
        clause_present = any(kw in lower for kw in rule.get("keywords", []))
        if not clause_present:
            missing_clauses.append(rule.get("name", "Unknown Clause"))
            if bool(rule.get("critical")):
                critical_missing += 1

    # STEP 4: Legal status decision logic.
    normalized_type = (doc_type or "").strip().lower()
    type_identified = normalized_type not in {"", "unknown", "not a legal document"}
    type_mismatch = False
    if ml_used and rule_label and rule_ratio >= 0.34 and ml_pred:
        type_mismatch = rule_label.strip().lower() != str(ml_pred).strip().lower()

    legal_status = "UNKNOWN"
    if type_identified and legal_structure_valid and confidence_pct > 65:
        legal_status = "LEGAL"
    elif type_mismatch or confidence_pct < 30 or no_legal_structure_detected:
        legal_status = "NOT LEGAL"
    elif partial_structure and len(missing_clauses) > 0:
        legal_status = "SUSPICIOUS"

    # STEP 5: Risk level.
    fake_pattern_detected, _ = classifier.detect_fraud(text)
    if fake_pattern_detected or critical_missing > 0 or len(missing_clauses) > 3:
        risk_level = "HIGH"
    elif 1 <= len(missing_clauses) <= 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    source_text = "rule-based" if (rule_label and rule_ratio > 0.60) else "ml-fallback"
    reasoning_summary = (
        f"Hybrid flow used {source_text}; structure checks passed: {structure_score}/7; "
        f"missing clauses: {len(missing_clauses)}."
    )

    return {
        "document_name": filename or "",
        "document_type": doc_type or "UNKNOWN",
        "legal_status": legal_status,
        "confidence_score": f"{int(max(0, min(100, confidence_pct)))}%",
        "risk_level": risk_level,
        "missing_clauses": missing_clauses,
        "reasoning_summary": reasoning_summary
    }
def extract_indian_entities(full_text: str) -> Dict[str, Any]:
    t = full_text or ""
    lower = t.lower()
    dates = []
    for m in re.finditer(DATE_PATTERN, t):
        raw = m.group(0)
        try:
            import dateparser
            parsed = dateparser.parse(raw)
            if parsed:
                dates.append(parsed.strftime("%d/%m/%Y"))
            else:
                dates.append(raw)
        except Exception:
            dates.append(raw)
    money = []
    for m in re.finditer(r"(₹|rs\.?)\s*[0-9,]+(?:\.\d+)?", t, flags=re.IGNORECASE):
        money.append(m.group(0))
    survey_numbers = []
    for m in re.finditer(r"(s\.?no\.?|survey\s*no\.?|ts\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE):
        survey_numbers.append(m.group(0))
    registration_numbers = []
    for m in re.finditer(r"(registration\s*no\.?|reg\.?\s*no\.?|doc\s*no\.?|stamp\s*no\.?|e-stamp\s*no\.?|grn)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE):
        registration_numbers.append(m.group(0))
    for m in re.finditer(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", t):
        registration_numbers.append("CIN: " + m.group(0))
    for m in re.finditer(r"\bDIN\s*[:\-]?\s*\d{8}\b", t, flags=re.IGNORECASE):
        registration_numbers.append(m.group(0))
    for m in re.finditer(r"\b(unique\s+doc(?:ument)?\s+ref(?:erence)?|uin|udrn)\b\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE):
        registration_numbers.append(m.group(0))
    for m in re.finditer(r"\bSRO\s*(?:code|name)?\b\s*[:\-]?\s*([A-Za-z0-9 \-]+)", t, flags=re.IGNORECASE):
        registration_numbers.append(m.group(0))
    case_numbers = []
    for m in re.finditer(r"(case\s*(no|number)|w\.?p\.?|c\.?r\.?p\.?|s\.?a\.?|suit\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE):
        case_numbers.append(m.group(0))
    courts = []
    for m in re.finditer(r"(supreme\s*court|high\s*court|district\s*court|magistrate\s*court|sessions\s*court|sub-registrar\s*office|registrar\s*office)", lower, flags=re.IGNORECASE):
        courts.append(m.group(0))
    land_types = []
    for k, pats in LAND_TYPE_PATTERNS.items():
        if any(p in lower for p in pats):
            land_types.append(k)
    reg_info = analyze_registration(t)
    parties = reg_info.get("parties") or []
    role_parties = []
    roles = ["vendor","vendee","buyer","seller","landlord","tenant","petitioner","respondent","employer","employee","authority","lessor","lessee","donor","donee"]
    for r in roles:
        if r in lower:
            role_parties.append(r)
    property_details = reg_info.get("property_schedule_excerpt")
    try:
        tamil_owner_blocks = []
        owner_patterns = [
            r"உரிமையாளர்(?:கள்)?\s*பெயர்[^\n]*\n([^\n]+)",
            r"உரிமையாளர்கள்\s*பெயர்[^\n]*\n([^\n]+)",
            r"உரிமையாளர்(?:கள்)?\s*[:\-]?\s*([^\n]+)",
            r"காரியதேயாளர்\s*[:\-]?\s*([^\n]+)",
        ]
        for pat in owner_patterns:
            for m in re.finditer(pat, t):
                tamil_owner_blocks.append(m.group(1))
        for block in tamil_owner_blocks:
            for nm in re.findall(r"([\u0B80-\u0BFF]{2,12}(?:\s+[\u0B80-\u0BFF]{2,12}){0,3})", block):
                parties.append(transliterate_tamil_to_latin(nm))
        if not tamil_owner_blocks:
            context_hits = ("தாலுகா" in t) or ("தாசில்தார்" in t) or ("கிராமம்" in t)
            if context_hits:
                for nm in re.findall(r"([\u0B80-\u0BFF]{3,12}(?:\s+[\u0B80-\u0BFF]{3,12}){0,2})", t):
                    parties.append(transliterate_tamil_to_latin(nm))
    except Exception:
        pass
    out = {
        "parties": parties,
        "role_parties": role_parties,
        "dates": dates,
        "money": money,
        "property_details": property_details or "",
        "survey_numbers": survey_numbers,
        "registration_numbers": registration_numbers,
        "case_numbers": case_numbers,
        "courts": courts,
        "land_types": land_types
    }
    return out
def validate_signers(full_text: str, signers: List[str]) -> Dict[str, Any]:
    # Sync fallback
    import asyncio
    try:
        return asyncio.run(validate_signers_async(full_text, signers))
    except Exception:
        return {"signers": [], "overall_signer_score": None}

def jurisdiction_specific_checks(full_text: str) -> Dict[str, Any]:
    # Sync fallback
    import asyncio
    try:
        return asyncio.run(jurisdiction_specific_checks_async(full_text))
    except Exception:
        return {"jurisdiction": None, "checks": [], "overall_compliance_score": None}

def build_timeline(full_text: str) -> List[Dict[str, Any]]:
    # Sync fallback
    import asyncio
    try:
        return asyncio.run(build_timeline_async(full_text))
    except Exception:
        return []

def curated_jurisdiction_templates(full_text: str) -> Dict[str, Any]:
    text = full_text.lower()
    checks = []
    juris = None
    if any(k in text for k in ["rera", "allottee", "promoter", "carpet area"]):
        juris = "India (Real Estate)"
        # RERA Checks
        rera_no = bool(re.search(r"rera\s*reg(istration)?\s*no\.?\s*[:\-]?\s*[a-z0-9\-]+", text))
        checks.append({"rule":"RERA Registration Number","status":"pass" if rera_no else "fail","detail":"Registration number presence","severity":"High"})
        
        promoter = "promoter" in text or "developer" in text or "builder" in text
        checks.append({"rule":"Promoter Details","status":"pass" if promoter else "fail","detail":"Promoter or developer named","severity":"Medium"})
        
        allottee = "allottee" in text or "purchaser" in text or "buyer" in text
        checks.append({"rule":"Allottee Details","status":"pass" if allottee else "fail","detail":"Allottee or buyer named","severity":"Medium"})
        
        carpet = "carpet area" in text or "square feet" in text or "sq.ft" in text
        checks.append({"rule":"Carpet Area Disclosure","status":"pass" if carpet else "fail","detail":"Carpet area specified","severity":"High"})
        
        possession = "possession" in text or "completion date" in text
        checks.append({"rule":"Possession Timeline","status":"pass" if possession else "fail","detail":"Delivery date mentioned","severity":"High"})

    elif any(k in text for k in ["gst", "pan", "tan", "income tax"]):
        juris = "India (Taxation)"
        gst_no = bool(re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", text.upper()))
        checks.append({"rule":"GSTIN Format","status":"pass" if gst_no else "fail","detail":"Valid GSTIN presence","severity":"High"})
        
        pan_no = bool(re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text.upper()))
        checks.append({"rule":"PAN Format","status":"pass" if pan_no else "fail","detail":"Valid PAN presence","severity":"Medium"})

    elif any(k in text for k in ["gdpr", "data protection", "personal data", "controller"]):
        juris = "EU (GDPR)"
        lawful = any(k in text for k in ["consent","contract","legitimate interests","legal obligation"])
        checks.append({"rule":"Lawful Basis","status":"pass" if lawful else "fail","detail":"GDPR lawful basis defined","severity":"High"})
        
        rights = any(k in text for k in ["right to access", "right to erase", "data portability"])
        checks.append({"rule":"Data Subject Rights","status":"pass" if rights else "fail","detail":"User rights specified","severity":"High"})
    
    else:
        juris = "Generic/Unknown"
        governing = "governing law" in text or "jurisdiction" in text
        checks.append({"rule":"Governing Law","status":"pass" if governing else "fail","detail":"Jurisdiction specified","severity":"Medium"})
    
    score = int(round(100 * (sum(1 for c in checks if c["status"]=="pass") / max(1,len(checks)))))
    return {"jurisdiction": juris, "checks": checks, "overall_compliance_score": score}

def analyze_registration(full_text: str) -> Dict[str, Any]:
    t = full_text
    reg_no = None
    m = re.search(r"(registration\s*no\.?|reg\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE)
    if m: reg_no = m.group(2)
    stamp_amount = None
    m2 = re.search(r"(stamp\s*duty|duty)\s*[:\-]?\s*(Rs\.?\s*[0-9,]+)", t, flags=re.IGNORECASE)
    if m2: stamp_amount = m2.group(2)
    office = None
    m3 = re.search(r"(sub-registrar|registrar)\s*office\s*[:\-]?\s*([A-Za-z\s,]+)", t, flags=re.IGNORECASE)
    if m3: office = m3.group(2).strip()
    property_sched = None
    ms = re.search(r"(schedule\s*property|property\s*schedule)\s*[:\-]?\s*(.+)", t, flags=re.IGNORECASE)
    if ms: property_sched = ms.group(2)[:400]
    parties = []
    try:
        nlp = _get_nlp()
        doc = nlp(full_text)
        parties = list({ent.text for ent in doc.ents if ent.label_ in ["PERSON","ORG"]})[:10]
    except Exception as e:
        print(f"[DEBUG] Registration Document NER failed: {e}")
    missing = []
    if not reg_no: missing.append("Registration Number")
    if not stamp_amount: missing.append("Stamp Duty")
    if not office: missing.append("Registrar Office")
    if not property_sched: missing.append("Property Schedule")
    return {
        "type": "Registration Document",
        "registration_number": reg_no,
        "stamp_duty": stamp_amount,
        "office": office,
        "property_schedule_excerpt": property_sched,
        "parties": parties,
        "missing_fields": missing
    }

from backend.app.services.risk_detection_service import risk_detection_service

def clause_risk_detection(full_text: str, translated_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced Hybrid Risk Detection (Rule-based + LegalBERT XAI)
    Segments document into clauses and analyzes each for risks.
    """
    text = (translated_text or full_text)
    
    # 1. Use advanced segmentation and LegalBERT analysis
    try:
        bert_risk_data = risk_detection_service.detect_all_risks(text)
        clauses = bert_risk_data.get("clauses", [])
        overall_risk = bert_risk_data.get("overall_risk", "Low")
        overall_score = bert_risk_data.get("overall_score", 0)
    except Exception as e:
        print(f"[DEBUG] RiskDetectionService failed: {e}")
        clauses = []
        overall_risk = "Low"
        overall_score = 0

    # 2. Rule-based augmentation (Ensuring specific legal keywords are caught)
    text_lower = text.lower()
    
    # If BERT found no clauses, fallback to basic rule-based snippets
    if not clauses:
        rules_clauses = []
        if "termination" in text_lower:
            rules_clauses.append({"clause": "Termination", "risk": "Medium", "risk_score": 50, "reason": "Termination clause detected via keywords."})
        if "indemnify" in text_lower:
            rules_clauses.append({"clause": "Indemnity", "risk": "High", "risk_score": 75, "reason": "Indemnity clause detected via keywords."})
        clauses = rules_clauses

    missing = [c for c in ["Termination", "Indemnity", "Liability", "Payment"] if c.lower() not in text_lower]

    return {
        "overall_risk": overall_risk,
        "overall_score": overall_score,
        "clauses": clauses,
        "missing_mandatory_fields": missing,
        "metadata": {
            "engine": "Hybrid (LegalBERT + Rules)",
            "version": "2.0.0"
        }
    }

async def clause_risk_detection_async(full_text: str, translated_text: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(clause_risk_detection, full_text, translated_text)

def llm_risk_validation(full_text: str) -> Dict[str, Any]:
    text = full_text or ""
    risk_data = clause_risk_detection(text)
    mand = check_mandatory_fields(text)
    curated = curated_jurisdiction_templates(text)
    issues = [c.get("reason") for c in (risk_data.get("clauses") or []) if (c or {}).get("risk") == "High" and (c or {}).get("reason")]
    for m in (mand.get("missing") or []):
        issues.append(f"Missing mandatory field: {m}")
    for chk in (curated.get("checks") or []):
        if (chk or {}).get("status") == "fail":
            rule_name = (chk or {}).get("rule") or "Unknown Check"
            issues.append(f"Jurisdiction check failed: {rule_name}")
    base_conf = 85
    if len(text.strip()) < 200:
        base_conf = max(50, base_conf - 15)
    base_conf = max(0, min(100, base_conf - min(20, 5 * len([x for x in issues if str(x).startswith('Missing mandatory field:')]))))
    level = risk_data.get("overall_risk") or "Unknown"
    if any((chk or {}).get("status") == "fail" for chk in (curated.get("checks") or [])) or len(mand.get("missing") or []) >= 3:
        level = "High"
    return {
        "risk_level": level,
        "issues_found": issues,
        "confidence_score": int(base_conf)
    }

async def llm_risk_validation_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(llm_risk_validation, full_text)

def check_mandatory_fields(full_text: str) -> Dict[str, Any]:
    """
    Module 9: Clause Presence Checker.
    Uses Hybrid (Rule-based + Semantic) logic to detect mandatory legal clauses.
    """
    try:
        # 1. Use advanced ClausePresenceService
        res = clause_presence_service.check_presence(full_text)
        
        return {
            "present": res["present"],
            "missing": res["missing"],
            "details": res["details"],
            "metadata": res["metadata"]
        }
    except Exception as e:
        print(f"[DEBUG] ClausePresenceService failed: {e}")
        # Fallback to basic keyword matching
        text = full_text.lower()
        fields = [
            ("Governing Law", any(k in text for k in ["governing law","jurisdiction"])),
            ("Signature", any(k in text for k in ["signature","signed by","authorized signatory"])),
            ("Effective Date", any(k in text for k in ["effective date","commencement"])),
            ("Parties", any(k in text for k in ["between","party","parties","agreement made by"])),
            ("Payment Terms", any(k in text for k in ["payment","consideration","fees","charges"])),
            ("Termination", "termination" in text),
            ("Confidentiality", "confidential" in text),
            ("Dispute Resolution", any(k in text for k in ["arbitration","court","dispute resolution","conciliation"])),
            ("Liability", "liability" in text),
            ("Warranty", "warranty" in text),
        ]
        return {
            "missing": [name for name, present in fields if not present], 
            "present": [name for name, present in fields if present],
            "metadata": {"engine": "Fallback-Rule-Based"}
        }

async def check_mandatory_fields_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(check_mandatory_fields, full_text)

def ask_openai_for_verification_and_confidence(text: str) -> Dict[str, Any]:
    # Rule-based verification
    # If text is long and has key clauses, we verify it.
    t = text.lower()
    has_clauses = any(k in t for k in ["agreement", "party", "signature", "signed"])
    marker = "Verified" if has_clauses else "UNVERIFIED"
    ai_confidence = 90 if has_clauses else 20
    return {
        "marker": marker,
        "ai_confidence": ai_confidence,
        "status": "LEGAL" if has_clauses else "UNVERIFIED",
        "confidence": ai_confidence,
        "note": "Verified using rule-based keyword density"
    }

async def ask_openai_for_verification_and_confidence_async(text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(ask_openai_for_verification_and_confidence, text)

async def analyze_registration_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(analyze_registration, full_text)

def compute_legality_score(analytics: Dict[str, Any]) -> float:
    """
    Computes a basic legality score based on presence of key components.
    """
    try:
        score = 0
        # 1. Signers present? (30 pts)
        sv = analytics.get("signer_validation", {})
        signer_score = sv.get("overall_signer_score") or 50
        score += signer_score * 0.3

        # 2. Compliance score? (30 pts)
        jc = analytics.get("jurisdiction_curated", {})
        compliance_score = jc.get("overall_compliance_score") or 50
        score += compliance_score * 0.3

        # 3. Risk clauses managed? (40 pts)
        risk_data = analytics.get("clause_risk", {})
        risk_score = risk_data.get("overall_risk_score") or 50
        score += max(0, 40 - (risk_score * 0.4))

        # 4. Mandatory fields (Bonus/Penalty)
        mand = analytics.get("mandatory_fields", {})
        missing = mand.get("missing_fields") or mand.get("missing") or []
        present = mand.get("present") or []

        if isinstance(missing, list) and (len(missing) > 0 or len(present) > 0):
            total_count = len(missing) + len(present)
            if total_count == 0:
                total_count = 5
            present_count = len(present)
            if present_count == 0 and len(missing) > 0:
                total_count = max(len(missing), 5)
                present_count = max(0, total_count - len(missing))
            score = (score * 0.8) + (20 * (present_count / max(1, total_count)))
        else:
            # If mandatory fields structure is missing, give partial credit
            score = score * 0.8 + 10

        return round(min(100, score), 2)
    except Exception as e:
        print(f"[DEBUG] compute_legality_score error: {e}")
        return 50  # Default score if calculation fails

def compute_ai_detection_score(text: str) -> Optional[int]:
    """
    Heuristic AI-generation likelihood score (0-100).
    Returns None if there's not enough text to make a meaningful guess.
    """
    try:
        if not text or not text.strip():
            return None
        words = re.findall(r"[A-Za-z']+", text)
        total = len(words)
        if total == 0:
            # Non-empty text with no alphabetic tokens (e.g., numbers/punctuation/code blobs)
            return 50
        if total < 3:
            # Keep tiny snippets as "insufficient", but score longer noisy/non-linguistic content.
            if len(text.strip()) <= 12:
                return None
            return 50
        unique = len(set(w.lower() for w in words))
        unique_ratio = unique / max(1, total)

        # Base score from lexical diversity (lower diversity => higher AI likelihood)
        if unique_ratio < 0.35:
            score = 70
        elif unique_ratio < 0.5:
            score = 55
        elif unique_ratio < 0.65:
            score = 35
        else:
            score = 20

        # Sentence length variance adjustment (very uniform sentences can indicate AI)
        sentences = [s.strip() for s in re.split(r"[\.!?]+", text) if s.strip()]
        if len(sentences) >= 5:
            lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
            avg = sum(lengths) / max(1, len(lengths))
            if avg > 0:
                variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
                stddev = math.sqrt(variance)
                cv = stddev / avg
                if cv < 0.4:
                    score += 10
                elif cv > 0.7:
                    score -= 10

        return int(max(0, min(100, round(score))))
    except Exception as e:
        print(f"[DEBUG] compute_ai_detection_score error: {e}")
        return 50  # Default score if calculation fails


def compute_genuineness_score(analytics: Dict[str, Any], ml_verification: Optional[Dict[str, Any]] = None) -> int:
    """
    Compute a practical genuineness score (0-100) by combining:
    - ML verification confidence
    - mandatory field coverage
    - clause risk (inverse contribution)
    - signer validation confidence
    """
    try:
        ml_verification = ml_verification or {}

        # 1) ML confidence (weight 40)
        ml_conf = ml_verification.get("ai_confidence")
        if ml_conf is None:
            ml_conf = analytics.get("ai_confidence", 0)
        ml_component = max(0, min(100, float(ml_conf or 0))) * 0.40

        # 2) Mandatory fields coverage (weight 20)
        mandatory = analytics.get("mandatory_fields", {}) or {}
        missing = mandatory.get("missing_fields") or mandatory.get("missing") or []
        present = mandatory.get("present") or []
        total_fields = len(missing) + len(present)
        if total_fields <= 0:
            mandatory_ratio = 0.5
        else:
            mandatory_ratio = len(present) / total_fields
        mandatory_component = (mandatory_ratio * 100.0) * 0.20

        # 3) Clause risk inverse (weight 25)
        clause_risk = analytics.get("clause_risk", {}) or {}
        risk_score = float(clause_risk.get("overall_risk_score", 50) or 50)
        risk_inverse = max(0.0, min(100.0, 100.0 - risk_score))
        risk_component = risk_inverse * 0.25

        # 4) Signer validation (weight 15)
        signer_validation = analytics.get("signer_validation", {}) or {}
        signer_score = signer_validation.get("overall_signer_score")
        if signer_score is None:
            signer_score = 50
        signer_component = max(0, min(100, float(signer_score))) * 0.15

        total = ml_component + mandatory_component + risk_component + signer_component
        return int(max(0, min(100, round(total))))
    except Exception as e:
        print(f"[DEBUG] compute_genuineness_score error: {e}")
        return 50  # Default score if calculation fails


def derive_legal_status(analytics: Dict[str, Any], ml_verification: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Determine final status for API consumers with property-document friendly rules.
    """
    analytics = analytics or {}
    ml_verification = ml_verification or {}
    marker = (ml_verification.get("marker") or analytics.get("verified_marker") or "").strip().lower()
    confidence = float(ml_verification.get("ai_confidence") or analytics.get("ai_confidence") or 0)
    doc_type = (analytics.get("document_type") or "Unknown").strip()
    legality = float(analytics.get("legality_score", 0) or 0)
    land = analytics.get("land_classification") or {}
    is_property_related = bool(land.get("is_property_related"))
    mandatory = analytics.get("mandatory_fields") or {}
    present_fields = set(mandatory.get("present") or [])
    missing_fields = set(mandatory.get("missing_fields") or mandatory.get("missing") or [])
    has_ownership_indicators = ("Ownership" in present_fields) or ("property_schedule_excerpt" in mandatory and bool(mandatory.get("property_schedule_excerpt")))
    property_types = {"Patta", "Sale Deed", "Land Property", "Property Certificate", "Settlement Deed", "Partition Deed", "Release Deed", "Rectification Deed"}

    # Empty analytics → Not legal (matches robustness tests)
    if not analytics:
        status = "NOT_LEGAL"
        reason = "Document failed verification thresholds"
    elif marker == "verified" and confidence >= 70:
        status = "LEGAL"
        reason = "Verified with high confidence"
    elif doc_type in property_types and (is_property_related or has_ownership_indicators):
        status = "LEGAL"
        reason = "Property document with key ownership indicators"
    elif marker in {"unverified", "rejected"}:
        _id_types = {
            "PAN Card",
            "e-PAN",
            "Aadhaar Card",
            "Aadhaar PVC Card",
            "e-Aadhaar",
            "Driving Licence",
            "Passport",
            "Voter ID",
            "Ration Card",
            "GST Certificate",
            "Birth Certificate",
            "Death Certificate",
            "Marriage Certificate",
            "Income Certificate",
            "Caste Certificate",
            "Domicile Certificate",
            "Salary Certificate",
            "Experience Certificate",
            "Bonafide Certificate",
            "Resume",
            "Invoice",
            "Bank Statement",
        }
        if doc_type in _id_types:
            status = "VALID"
            reason = "Identity/certificate document — valid by type"
        else:
            status = "NOT_LEGAL"
            reason = "Document failed verification thresholds"
    else:
        # Fall back to legality score if strong evidence is present
        if legality >= 55:
            status = "LEGAL"
            reason = "Score-based legal determination"
        elif legality >= 40:
            status = "NEEDS_REVIEW"
            reason = "Borderline score; needs manual review"
        else:
            status = "UNKNOWN"
            reason = "Insufficient evidence"

    summary = (
        f"Status: {status}. Type: {doc_type}. "
        f"Confidence: {confidence:.1f}%. Legality score: {legality:.1f}/100."
    )
    return {"status": status, "reason": reason, "summary": summary}


def build_required_output_schema(results: Dict[str, Any], analytics: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Build strict response contract requested by frontend/product:
    document_type, summary_simple, entities, clauses, key_terms,
    risk_analysis, compliance_flags, confidence_score
    """
    entities_block = {
        "names": ((results or {}).get("entities") or {}).get("names", []),
        "organizations": ((results or {}).get("entities") or {}).get("organizations", []),
        "emails": ((results or {}).get("contact_info") or {}).get("emails", []),
        "phones": ((results or {}).get("contact_info") or {}).get("phones", []),
        "dates": (results or {}).get("dates", []),
        "survey_numbers": (((results or {}).get("metadata") or {}).get("survey_numbers") or []),
        "case_numbers": (((results or {}).get("metadata") or {}).get("case_numbers") or []),
        "authorities": (((results or {}).get("metadata") or {}).get("authorities") or []),
    }

    clause_counts = (results or {}).get("clauses_found", {}) or {}
    clauses = [k for k, v in clause_counts.items() if int(v or 0) > 0]
    clause_risk_items = (((analytics or {}).get("clause_risk") or {}).get("clauses") or [])
    for c in clause_risk_items:
        c_name = (c or {}).get("clause")
        if c_name and c_name not in clauses:
            clauses.append(c_name)

    key_terms = list((((results or {}).get("metadata") or {}).get("keyword_frequency") or {}).keys())[:12]

    risk_analysis = {
        "overall_risk": (((analytics or {}).get("clause_risk") or {}).get("overall_risk") or "Unknown"),
        "overall_risk_score": (((analytics or {}).get("clause_risk") or {}).get("overall_risk_score")),
        "combined_risk_index": (analytics or {}).get("combined_risk_index"),
        "legal_status": (analytics or {}).get("legal_status"),
        "high_risk_reasons": [
            (x or {}).get("reason")
            for x in clause_risk_items
            if (x or {}).get("risk") == "High" and (x or {}).get("reason")
        ][:8],
    }

    compliance_flags: List[Dict[str, Any]] = []
    for field in (((analytics or {}).get("mandatory_fields") or {}).get("missing") or []):
        compliance_flags.append({"type": "mandatory_field_missing", "field": field, "severity": "Medium"})
    for field in (((analytics or {}).get("registration_analysis") or {}).get("missing_fields") or []):
        compliance_flags.append({"type": "registration_gap", "field": field, "severity": "High"})
    for chk in (((analytics or {}).get("jurisdiction_curated") or {}).get("checks") or []):
        if (chk or {}).get("status") == "fail":
            compliance_flags.append({
                "type": "jurisdiction_check_failed",
                "field": (chk or {}).get("rule"),
                "severity": (chk or {}).get("severity", "Medium")
            })

    summary_simple = (analytics or {}).get("status_summary") or (results or {}).get("summary") or summarize_fallback(text or "")

    return {
        "document_type": (analytics or {}).get("document_type") or "Generic Legal Document",
        "summary_simple": summary_simple,
        "entities": entities_block,
        "clauses": clauses,
        "key_terms": key_terms,
        "risk_analysis": risk_analysis,
        "compliance_flags": compliance_flags,
        "confidence_score": (analytics or {}).get("genuineness_score") or (analytics or {}).get("ai_confidence") or 0,
        "land_classification": (analytics or {}).get("land_classification") or (((results or {}).get("metadata") or {}).get("land_classification"))
    }
