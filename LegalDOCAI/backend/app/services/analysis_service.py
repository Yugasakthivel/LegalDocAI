import re
import time
import json
import random
import math
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
from typing import Tuple

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
    "Punjai": ["punjai", "dry land", "dryland"],
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
    Summarize text using frequency-based extraction (Rule-Based).
    """
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 30]
        if not sents:
             return "Text too short to summarize."
             
        # Keyword frequency
        freq = Counter(t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and t.is_alpha)
        maxf = max(freq.values()) if freq else 1
        
        sent_scores = {}
        for s in sents:
            s_doc = nlp(s.lower())
            score = sum(freq.get(t.lemma_.lower(),0)/maxf for t in s_doc)
            # Normalize by length to avoid favoring long sentences
            sent_scores[s] = score / max(1, len(s_doc))
            
        # Pick top 3
        top_sents = sorted(sent_scores, key=sent_scores.get, reverse=True)[:3]
        return " ".join(top_sents)
    else:
        # Fallback without SpaCy
        return text[:500] + "..."

def answer_with_fallback(question: str, context: str) -> str:
    # Rule-based QA (Simple Keyword Search)
    q_tokens = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "of", "in", "a", "an", "to"}
    if not q_tokens: return "Could not understand question."
    
    # Split context into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", context)
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
        indian = extract_indian_entities(full_text)
    except Exception:
        indian = {"parties": [], "role_parties": [], "registration_numbers": [], "survey_numbers": [], "case_numbers": [], "courts": [], "land_types": [], "dates": [], "money": [], "property_details": "", "role_parties": []}
    if indian:
        names = sanitize_names((names or []) + (indian.get("parties") or []) + (indian.get("role_parties") or []))
    
    emails = reg_results["emails"]
    phone_details = reg_results["phone_details"]
    monetary_values = reg_results["monetary_values"]
    survey_numbers = reg_results["survey_numbers"]
    case_numbers = reg_results["case_numbers"]
    authorities = reg_results["authorities"]
    dates = reg_results["dates"]

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
    land_classification = extract_land_classification(full_text)

    # Clause Detection
    search_text = working_text if len(working_text) > 100 else full_text
    clause_counter = Counter()
    for kw in CLAUSE_KEYWORDS:
        clause_counter[kw] = len(re.findall(r"\b" + re.escape(kw) + r"\b", search_text, flags=re.IGNORECASE))
    
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
    nlp = _get_nlp()
    sents: List[str] = []
    if nlp:
        def _get_sents(text):
            doc = nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        sents = await asyncio.to_thread(_get_sents, full_text[:10000])
    else:
        sents = [x.strip() for x in re.split(r"(?<=[\.\!\?\n])\s+", full_text) if x.strip()]
    
    events: List[Dict[str, Any]] = []
    for sent in sents:
        found = re.findall(DATE_PATTERN, sent)
        for d in found:
            parsed = dateparser.parse(d)
            if parsed:
                events.append({"date": str(parsed.date()), "event": sent})
    
    if not events and is_openai_ready():
        try:
            prompt = (
                "Extract a timeline from the legal document. "
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
            # Extract JSON array
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1:
                events = json.loads(raw[start:end])
        except Exception as e:
            print(f"[DEBUG] build_timeline_async error: {e}")
            try:
                msg = str(e).lower()
                if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
                    mark_quota_exhausted()
            except Exception:
                pass
            
    return sorted(events, key=lambda x: x.get("date", ""))[:15]

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

        prediction, confidence = classifier.predict_document_type(full_text)

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
        "Patta","Chitta","TSLR","FMB Sketch","Layout Approval","Sale Deed","Settlement Deed",
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
    if doc_type in {"Resume","Invoice","ID Card"}:
        return "Non-Legal Document"
    return "Other"

async def classify_document_category_async(full_text: str) -> str:
    return await asyncio.to_thread(classify_document_category, full_text)
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
    except Exception: pass
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

def clause_risk_detection(full_text: str, translated_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced Rule-Based Risk Detection (Replaces LLM)
    Detects specific high-risk patterns in legal text.
    """
    text = (translated_text or full_text).lower()
    
    clauses = []
    
    # 1. Termination Risks
    # Pattern: "without cause" or "convenience" is usually high risk for the other party
    term_risk = "Low"
    term_reason = "Standard termination clause."
    if "termination" in text or "terminate" in text:
        if "without cause" in text or "for convenience" in text:
            term_risk = "High"
            term_reason = "Clause allows termination 'without cause' or 'for convenience'."
        elif "immediate effect" in text:
            term_risk = "Medium"
            term_reason = "Termination with immediate effect found."
        
        snippet_idx = text.find("termination")
        snippet = full_text[snippet_idx:snippet_idx+150] + "..." if snippet_idx != -1 else ""
        clauses.append({"clause": "Termination", "present": True, "risk": term_risk, "risk_score": 80 if term_risk=="High" else (50 if term_risk=="Medium" else 20), "reason": term_reason, "highlight": snippet})
    else:
        clauses.append({"clause": "Termination", "present": False, "risk": "Medium", "risk_score": 50, "reason": "No termination clause found.", "highlight": ""})

    # 2. Indemnity
    # Pattern: "indemnify" without "cap" or "limit" is risky
    if "indemnify" in text or "indemnification" in text:
        risk = "Medium"
        reason = "Indemnity clause present."
        if "unlimited" in text or "all claims" in text:
            risk = "High"
            reason = "Potential unlimited indemnity liability."
        elif "cap" in text or "limit" in text:
            risk = "Low"
            reason = "Indemnity appears to be capped/limited."
            
        clauses.append({"clause": "Indemnity", "present": True, "risk": risk, "risk_score": 75 if risk=="High" else 30, "reason": reason, "highlight": "Indemnity clause detected..."})
    else:
        clauses.append({"clause": "Indemnity", "present": False, "risk": "Low", "risk_score": 10, "reason": "No indemnity clause (Risk depends on side).", "highlight": ""})
        
    # 3. Dispute Resolution
    if "arbitration" in text:
        clauses.append({"clause": "Dispute Resolution", "present": True, "risk": "Low", "risk_score": 20, "reason": "Arbitration clause present.", "highlight": "Arbitration..."})
    elif "court" in text or "jurisdiction" in text:
        clauses.append({"clause": "Dispute Resolution", "present": True, "risk": "Medium", "risk_score": 40, "reason": "Litigation/Court jurisdiction specified.", "highlight": "Court jurisdiction..."})
    else:
        clauses.append({"clause": "Dispute Resolution", "present": False, "risk": "High", "risk_score": 90, "reason": "No dispute resolution mechanism defines.", "highlight": ""})

    # 4. Liability
    if "limitation of liability" in text or "limit liability" in text:
        clauses.append({"clause": "Liability", "present": True, "risk": "Low", "risk_score": 20, "reason": "Liability limitation present.", "highlight": "Limitation of Liability..."})
    else:
        clauses.append({"clause": "Liability", "present": False, "risk": "High", "risk_score": 85, "reason": "No limitation of liability found (High Risk).", "highlight": ""})

    # 5. Payment
    pay_present = any(k in text for k in ["payment","consideration","fees","rent"])
    pay_risk = "Low"
    pay_reason = "Payment terms present."
    if pay_present:
        if "late fee" in text or "interest" in text or "penalty" in text:
            pay_risk = "Medium"
            pay_reason = "Late fee/interest/penalty conditions found."
        if "advance" in text and "non-refundable" in text:
            pay_risk = "High"
            pay_reason = "Non-refundable advance detected."
        clauses.append({"clause":"Payment","present":True,"risk":pay_risk,"risk_score":70 if pay_risk=="High" else (40 if pay_risk=="Medium" else 20),"reason":pay_reason,"highlight":""})
    else:
        clauses.append({"clause":"Payment","present":False,"risk":"High","risk_score":85,"reason":"No clear payment/consideration terms.","highlight":""})

    # 6. Ownership
    own_present = any(k in text for k in ["ownership","title","transfer","conveyance"])
    own_risk = "Low" if own_present else "High"
    own_reason = "Ownership or title statements present." if own_present else "Ownership/title statements missing."
    clauses.append({"clause":"Ownership","present":own_present,"risk":own_risk,"risk_score":80 if own_risk=="High" else 25,"reason":own_reason,"highlight":""})

    # 7. Jurisdiction/Governing Law
    juris_present = any(k in text for k in ["governing law","jurisdiction","court"])
    juris_risk = "Low" if juris_present else "Medium"
    juris_reason = "Jurisdiction/governing law specified." if juris_present else "Jurisdiction/governing law missing."
    clauses.append({"clause":"Jurisdiction","present":juris_present,"risk":juris_risk,"risk_score":60 if juris_risk=="Medium" else 20,"reason":juris_reason,"highlight":""})

    # 8. Rights & Obligations
    ro_present = any(k in text for k in ["obligation","obligations","rights","duty"])
    ro_risk = "Low" if ro_present else "Medium"
    ro_reason = "Rights/obligations described." if ro_present else "Rights/obligations unclear."
    clauses.append({"clause":"Rights & Obligations","present":ro_present,"risk":ro_risk,"risk_score":55 if ro_risk=="Medium" else 20,"reason":ro_reason,"highlight":""})

    # 9. Penalties
    pen_present = any(k in text for k in ["penalty","fine","liquidated damages","ld"])
    pen_risk = "Medium" if pen_present else "Low"
    pen_reason = "Penalty/LD present." if pen_present else "No explicit penalties."
    clauses.append({"clause":"Penalties","present":pen_present,"risk":pen_risk,"risk_score":45 if pen_risk=="Medium" else 15,"reason":pen_reason,"highlight":""})

    # Calculate Overall
    total_score = sum(c["risk_score"] for c in clauses)
    avg_score = total_score / max(1, len(clauses))
    
    overall_risk = "Low"
    if avg_score > 40: overall_risk = "Medium"
    if avg_score > 70: overall_risk = "High"
    
    missing = [c["clause"] for c in clauses if not c["present"]]

    return {
        "clauses": clauses,
        "overall_risk": overall_risk, 
        "overall_risk_score": int(avg_score), 
        "jurisdiction": "Inferred (Rule-Based)", 
        "missing_mandatory_fields": missing
    }

async def clause_risk_detection_async(full_text: str, translated_text: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(clause_risk_detection, full_text, translated_text)

def llm_risk_validation(full_text: str) -> Dict[str, Any]:
    """
    Delegate to Regex risk detection (Mocking the LLM validation structure)
    """
    risk_data = clause_risk_detection(full_text)
    return {
        "risk_level": risk_data["overall_risk"],
        "issues_found": [c["reason"] for c in risk_data["clauses"] if c["risk"] == "High"],
        "confidence_score": 85 # Mock confidence for rule-based
    }

async def llm_risk_validation_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(llm_risk_validation, full_text)

def check_mandatory_fields(full_text: str) -> Dict[str, Any]:
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
    return {"missing": [name for name, present in fields if not present], "present": [name for name, present in fields if present]}

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
    property_types = {"Patta", "Sale Deed", "Property Certificate", "Settlement Deed", "Partition Deed", "Release Deed", "Rectification Deed"}

    if marker == "verified" and confidence >= 70:
        status = "LEGAL"
        reason = "Verified with high confidence"
    elif doc_type in property_types and (is_property_related or has_ownership_indicators):
        status = "LEGAL"
        reason = "Property document with key ownership indicators"
    elif marker in {"unverified", "rejected"}:
        status = "NEEDS_REVIEW"
        reason = "Document unverified"
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
