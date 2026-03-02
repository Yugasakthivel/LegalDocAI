import re
import json
import asyncio
from typing import Dict, Any, List
from backend.app.services.nlp_service import _get_nlp

from backend.app.services.compliance_checker import compliance_service

def jurisdiction_checks(full_text: str) -> Dict[str, Any]:
    """
    Module 10: Regulatory Compliance Checker.
    Automatically verify whether a legal document complies with regulatory requirements.
    """
    try:
        # 1. Use advanced ComplianceService (Hybrid Rule + Semantic)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(compliance_service.check_compliance(full_text))
        loop.close()
        
        # 2. Map to existing structure for backward compatibility
        checks = []
        for violation in res["violations"]:
            checks.append({
                "rule": violation,
                "status": "fail",
                "detail": "Regulatory requirement not met.",
                "severity": "High" if res["risk_level"] == "High" else "Medium"
            })
            
        return {
            "jurisdiction": "India (Inferred)",
            "checks": checks,
            "overall_compliance_score": res["compliance_score"],
            "risk_level": res["risk_level"],
            "recommendations": res["recommendations"],
            "metadata": res["metadata"]
        }
    except Exception as e:
        print(f"[DEBUG] ComplianceService failed: {e}")
        # Original Rule-based Fallback
        text = full_text.lower()
        jurisdiction = "India" if any(k in text for k in ["india", "rs.", "rupees"]) else "USA"
        checks = []
        if jurisdiction == "India":
            checks.append({"rule": "Stamp Duty", "status": "pass" if "stamp duty" in text else "fail", "severity": "High"})
            checks.append({"rule": "Registration", "status": "pass" if "registration" in text else "fail", "severity": "High"})
        else:
            checks.append({"rule": "Governing Law", "status": "pass" if "governing law" in text else "fail", "severity": "Medium"})
        
        score = 100 if all(c["status"] == "pass" for c in checks) else 50
        return {
            "jurisdiction": jurisdiction,
            "checks": checks,
            "overall_compliance_score": score
        }

async def jurisdiction_checks_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(jurisdiction_checks, full_text)

def curated_checks(full_text: str) -> Dict[str, Any]:
    try:
        from backend.app.services.analysis_service import curated_jurisdiction_templates
        return curated_jurisdiction_templates(full_text)
    except Exception:
        return jurisdiction_checks(full_text)

async def curated_checks_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(curated_checks, full_text)

def mandatory_fields(full_text: str, doc_type: str = None) -> Dict[str, Any]:
    # Regex-based extraction helper
    t = full_text
    doc_type_lower = (doc_type or "").lower()
    tl = t.lower()
    
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
    if "schedule" in tl and "property" in tl:
        property_sched = "Property Schedule detected (excerpt hidden for brevity)"
        
    parties = []
    # Try basic regex for parties if NLP fails or is too heavy
    if not parties:
        # Naive "Between X and Y"
        m_parties = re.search(r"between\s+([A-Za-z\s\.,]+)\s+(and|&)\s+([A-Za-z\s\.,]+)", t, flags=re.IGNORECASE)
        if m_parties:
            parties = [m_parties.group(1).strip(), m_parties.group(3).strip()]

    missing_fields = []
    property_types = [
        "sale deed", "lease deed", "rental agreement", "gift deed",
        "patta", "tslr", "encumbrance certificate", "land property"
    ]
    is_property_doc = any(k in doc_type_lower for k in property_types)
    stamp_duty_present = "stamp duty" in tl
    registration_present = bool(re.search(r"(registration\s*no|reg\s*no)", t, re.I))
    sub_registrar_present = ("sub-registrar" in tl) or ("sub registrar" in tl)
    property_desc_present = ("schedule property" in tl) or ("property schedule" in tl)
    document_category = "Generic"
    court_types = ["court judgment", "court order", "legal notice", "affidavit", "petition"]
    business_types = ["corporate agreement", "partnership deed", "moa", "aoa", "regulatory filing", "legal agreement", "contract"]
    personal_cert_types = ["birth certificate", "death certificate", "marriage certificate", "caste certificate", "income certificate", "domicile certificate", "disability certificate", "residence certificate", "bonafide certificate", "migration certificate", "legal heir certificate", "puc certificate", "transfer certificate"]
    employment_types = ["employment contract", "employee id", "experience certificate", "salary certificate"]
    is_court_doc = any(k in doc_type_lower for k in court_types)
    is_business_doc = any(k in doc_type_lower for k in business_types)
    is_personal_cert_doc = any(k in doc_type_lower for k in personal_cert_types)
    is_employment_doc = any(k in doc_type_lower for k in employment_types)
    if is_property_doc:
        document_category = "Property"
        if not (stamp_duty_present or stamp_amount): missing_fields.append("Stamp Duty")
        if not (registration_present or reg_no): missing_fields.append("Registration Number")
        if not (sub_registrar_present or office): missing_fields.append("Registrar Office")
        if not (property_desc_present or property_sched): missing_fields.append("Property Description/Schedule")
    elif is_court_doc:
        document_category = "Court"
        if not re.search(r"\bcase\s*no\.?\b", t, flags=re.IGNORECASE): missing_fields.append("Case Number")
        if not re.search(r"\bcourt\b", t, flags=re.IGNORECASE): missing_fields.append("Court Name")
        if not re.search(r"\bjudge|justice\b", t, flags=re.IGNORECASE): missing_fields.append("Judge")
        if not re.search(r"\bpetitioner|appellant|plaintiff\b", t, flags=re.IGNORECASE): missing_fields.append("Petitioner/Appellant")
        if not re.search(r"\brespondent|defendant\b", t, flags=re.IGNORECASE): missing_fields.append("Respondent/Defendant")
    elif is_business_doc:
        document_category = "Business"
        if not re.search(r"\bcompany\s*name|firm\s*name\b", t, flags=re.IGNORECASE): missing_fields.append("Company/Firm Name")
        if not re.search(r"\bCIN|GSTIN\b", t, flags=re.IGNORECASE): missing_fields.append("CIN/GSTIN")
        if not re.search(r"\bregistered\s*office|office\s*address\b", t, flags=re.IGNORECASE): missing_fields.append("Registered Office Address")
        if not re.search(r"\bdirector|partner\b", t, flags=re.IGNORECASE): missing_fields.append("Authorized Signatory")
    elif is_personal_cert_doc:
        document_category = "Personal Certificate"
        if not re.search(r"\bcertificate\s*no\.?\b", t, flags=re.IGNORECASE): missing_fields.append("Certificate Number")
        if not re.search(r"\bissued\s*by|authority\b", t, flags=re.IGNORECASE): missing_fields.append("Issuing Authority")
        if not re.search(r"\bdate\s*of\s*issue\b", t, flags=re.IGNORECASE): missing_fields.append("Date of Issue")
        if not re.search(r"\bname\b", t, flags=re.IGNORECASE): missing_fields.append("Holder Name")
    elif is_employment_doc:
        document_category = "Employment"
        if not re.search(r"\bemployer|company\b", t, flags=re.IGNORECASE): missing_fields.append("Employer")
        if not re.search(r"\bdesignation|role\b", t, flags=re.IGNORECASE): missing_fields.append("Designation/Role")
        if not re.search(r"\bdate\s*of\s*joining|salary\b", t, flags=re.IGNORECASE): missing_fields.append("Joining Date/Salary")
    else:
        document_category = "Generic"
        if not reg_no: missing_fields.append("Registration Number")
        if not stamp_amount: missing_fields.append("Stamp Duty")
        if not office: missing_fields.append("Registrar Office")
    
    return {
        "type": "Legal Document",
        "document_category": document_category,
        "registration_number": reg_no,
        "stamp_duty": stamp_amount,
        "office": office,
        "property_schedule_excerpt": property_sched,
        "parties": parties,
        "missing_fields": missing_fields
    }

async def mandatory_fields_async(full_text: str, doc_type: str = None) -> Dict[str, Any]:
    return await asyncio.to_thread(mandatory_fields, full_text, doc_type)
