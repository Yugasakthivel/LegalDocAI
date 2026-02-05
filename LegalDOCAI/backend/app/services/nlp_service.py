import re
import time
import spacy
from typing import List, Dict, Any, Optional
from collections import Counter
import dateparser
import json

# Constants
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"


# Constants
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"\+?\d[\d\s-]{7,}\d"
DATE_PATTERN = r"(?i)\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b"
CLAUSE_KEYWORDS = [
    "termination", "confidentiality", "liability", "warranty",
    "dispute", "governing law", "payment", "obligation",
    "indemnity", "agreement"
]

try:
    import phonenumbers
except ImportError:
    phonenumbers = None

_nlp: Optional[spacy.language.Language] = None

def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback or try to download? usually better to fail gracefully or use blank
            # For this environment, we assume it's installed or we handle it
            pass
    return _nlp

def sanitize_names(raw: List[str]) -> List[str]:
    cleaned = []
    for n in raw:
        s = (n or "").strip()
        if len(s) < 3 or len(s) > 60: continue
        if re.search(r"\d", s): continue
        tokens = [t for t in s.split() if any(ch.isalpha() for ch in t)]
        if not tokens: continue
        if len(tokens) > 6: continue
        joined = " ".join(tokens)
        letters = len(re.findall(r"[A-Za-z]", joined))
        total = len(joined)
        if total == 0 or letters/total < 0.7: continue
        if len(re.findall(r"[aeiouAEIOU]", joined)) == 0: continue
        if len(re.findall(r"[^A-Za-z\s\.\-']", joined)) > max(1, total//10): continue
        cleaned.append(joined)
    seen = set()
    out = []
    for x in cleaned:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def sanitize_orgs(raw: List[str]) -> List[str]:
    cleaned = []
    for n in raw:
        s = (n or "").strip()
        if len(s) < 3 or len(s) > 80: continue
        tokens = [t for t in s.split() if any(ch.isalpha() for ch in t)]
        if not tokens: continue
        if len(tokens) > 8: continue
        joined = " ".join(tokens)
        letters = len(re.findall(r"[A-Za-z]", joined))
        total = len(joined)
        if total == 0 or letters/total < 0.6: continue
        if len(re.findall(r"[^A-Za-z0-9\s\.\-&']", joined)) > max(1, total//10): continue
        cleaned.append(joined)
    seen = set()
    out = []
    for x in cleaned:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def extract_signatures(text: str) -> List[str]:
    found = []
    for m in re.finditer(r"(signed by|signature|authorized signatory|attested by)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)", text, flags=re.IGNORECASE):
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
        except Exception: pass
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
            upper_ratio = sum(1 for c in l if c.isupper())/max(1,len(re.sub(r"[^A-Za-z]","",l)))
            if upper_ratio > 0.6 or re.match(r"^\d+(\.\d+)*\s", l) or re.match(r"^\([a-zA-Z]\)\s", l):
                if l not in heads: heads.append(l)
    return heads

def build_timeline(full_text: str) -> List[Dict[str, Any]]:
    nlp = _get_nlp()
    sents: List[str] = []
    try:
        if nlp:
            doc = nlp(full_text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            sents = [x.strip() for x in re.split(r"(?<=[\.\!\?\n])\s+", full_text) if x.strip()]
    except Exception:
        sents = [x.strip() for x in re.split(r"(?<=[\.\!\?\n])\s+", full_text) if x.strip()]
    events: List[Dict[str, Any]] = []
    for sent in sents:
        found = re.findall(DATE_PATTERN, sent)
        for d in found:
            try:
                parsed = dateparser.parse(d)
            except Exception:
                parsed = None
            if parsed:
                events.append({"date": str(parsed.date()), "event": sent})
    if not events:
        # Fallback to simple sentence scanning if regex failed (already covered above or leave empty)
        # OpenAI fallback removed.
        pass
    events.sort(key=lambda x: x["date"])
    return events

def classify_document_type(full_text: str) -> str:
    from backend.app.services.analysis_service import classify_document_type as _classify
    return _classify(full_text)

def analyze_text(full_text: str):
    nlp = _get_nlp()
    if nlp:
        doc = nlp(full_text)
        names = sanitize_names([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
        orgs = sanitize_orgs([ent.text for ent in doc.ents if ent.label_ == "ORG"])
        
        # Summarization
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        summary = "No summary available."
        if sentences:
            freq = Counter(t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct)
            maxf = max(freq.values()) if freq else 1
            sent_scores = {}
            for s in sentences:
                s_doc = nlp(s.lower())
                score = sum(freq.get(t.lemma_.lower(),0)/maxf for t in s_doc)
                sent_scores[s] = score
            summary = " ".join(s for s,_ in sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:3])
        
        tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
        keyword_frequency = dict(Counter(tokens).most_common(10))
    else:
        # Fallback if spaCy is not loaded
        names = []
        orgs = []
        summary = "NLP model not loaded."
        keyword_frequency = {}

    emails = re.findall(EMAIL_PATTERN, full_text)
    phones = extract_phones_validated(full_text)
    phone_details = []
    for s in phones:
        digits = re.sub(r"\D", "", s or "")
        normalized = f"+91 {digits}" if len(digits) == 10 and digits[:1] in "6789" else s
        tag = phone_compliance_tag(s)
        phone_details.append({"number": normalized, "normalized": normalized, "tag": tag})
        
    dates = [str(dateparser.parse(d).date()) for d in re.findall(DATE_PATTERN, full_text) if dateparser.parse(d)]
    
    clause_counter = Counter()
    clause_timings = {}
    for kw in CLAUSE_KEYWORDS:
        t0 = time.perf_counter()
        clause_counter[kw] = len(re.findall(r"\b" + re.escape(kw) + r"\b", full_text, flags=re.IGNORECASE))
        t1 = time.perf_counter()
        clause_timings[kw] = int((t1 - t0) * 1000)
        
    signers = sanitize_names(extract_signatures(full_text))
    
    total_clauses = sum(clause_counter.values())
    legality_score = min(100, int(
        min(40, len(set(names))*2) +
        min(30, total_clauses*4) +
        min(30, (len(set(emails))+len(set(phones)))*2)
    ))

    results_summary = {
        "names": list(dict.fromkeys(names)),
        "organizations": list(dict.fromkeys(orgs)),
        "emails": list(set(emails)),
        "phones": list(set(phones)),
        "phone_details": phone_details,
        "dates": list(set(dates)),
        "clauses_found": list(clause_counter.keys()),
        "signers": list(set(signers))
    }

    analytics = {
        "clause_summary": dict(clause_counter),
        "clause_scan_timings_ms": clause_timings,
        "keyword_frequency": keyword_frequency,
        "summary": summary,
        "legality_score": legality_score,
        "clause_headings": detect_clause_headings(full_text),
        "total_names": len(results_summary["names"]),
        "total_emails": len(results_summary["emails"]),
        "total_phones": len(results_summary["phones"]),
        "total_signers": len(results_summary["signers"]),
        "total_clauses": total_clauses,
        "timeline": build_timeline(full_text),
        "document_type": classify_document_type(full_text)
    }

    return results_summary, analytics
