import re
import json
import asyncio
from typing import Dict, Any, List
from backend.app.services.nlp_service import _get_nlp

def jurisdiction_checks(full_text: str) -> Dict[str, Any]:
    """
    Rule-based jurisdiction checks replacing LLM logic.
    """
    text = full_text.lower()
    
    # Simple Inference
    jurisdiction = "Unknown"
    if "india" in text or "rs." in text or "rupees" in text:
        jurisdiction = "India"
    elif "united states" in text or "usa" in text or "dollar" in text:
        jurisdiction = "USA"
    
    checks = []
    
    if jurisdiction == "India":
        checks.append({
            "rule": "Stamp Duty Payment",
            "status": "pass" if "stamp duty" in text else "fail",
            "detail": "Check for mention of Stamp Duty payment.",
            "severity": "High"
        })
        checks.append({
            "rule": "Registration",
            "status": "pass" if "registration" in text or "registered" in text else "fail",
            "detail": "Check if document is registered.",
            "severity": "High"
        })
    else:
        # Generic Checks
        checks.append({
            "rule": "Governing Law",
            "status": "pass" if "governing law" in text else "fail",
            "detail": "Governing law clause presence.",
            "severity": "Medium"
        })

    score = 100 if all(c["status"] == "pass" for c in checks) else 50
    if not checks: score = 0

    return {
        "jurisdiction": jurisdiction,
        "checks": checks,
        "overall_compliance_score": score
    }

async def jurisdiction_checks_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(jurisdiction_checks, full_text)

def curated_checks(full_text: str) -> Dict[str, Any]:
    return jurisdiction_checks(full_text)

async def curated_checks_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(curated_checks, full_text)

def mandatory_fields(full_text: str) -> Dict[str, Any]:
    # Regex-based extraction helper
    t = full_text
    
    # 1. Registration Number
    reg_no = None
    m = re.search(r"(registration\s*no\.?|reg\.?\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.IGNORECASE)
    if m: reg_no = m.group(2)
    
    # 2. Stamp Duty
    stamp_amount = None
    m2 = re.search(r"(stamp\s*duty|duty)\s*[:\-]?\s*(Rs\.?\s*[0-9,]+)", t, flags=re.IGNORECASE)
    if m2: stamp_amount = m2.group(2)
    
    # 3. Office
    office = None
    m3 = re.search(r"(sub-registrar|registrar)\s*office\s*[:\-]?\s*([A-Za-z\s,]+)", t, flags=re.IGNORECASE)
    if m3: office = m3.group(2).strip()
    
    # 4. Property Schedule
    property_sched = None
    if "schedule" in t.lower() and "property" in t.lower():
        property_sched = "Property Schedule detected (excerpt hidden for brevity)"
        
    parties = []
    # Try basic regex for parties if NLP fails or is too heavy
    if not parties:
        # Naive "Between X and Y"
        m_parties = re.search(r"between\s+([A-Za-z\s\.,]+)\s+(and|&)\s+([A-Za-z\s\.,]+)", t, flags=re.IGNORECASE)
        if m_parties:
            parties = [m_parties.group(1).strip(), m_parties.group(3).strip()]

    missing_fields = []
    if not reg_no: missing_fields.append("Registration Number")
    if not stamp_amount: missing_fields.append("Stamp Duty")
    if not office: missing_fields.append("Registrar Office")
    
    return {
        "type": "Legal Document",
        "registration_number": reg_no,
        "stamp_duty": stamp_amount,
        "office": office,
        "property_schedule_excerpt": property_sched,
        "parties": parties,
        "missing_fields": missing_fields
    }

async def mandatory_fields_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(mandatory_fields, full_text)
