import re
import difflib
import logging
from typing import Dict, Any, List, Tuple
from backend.app.services.analysis_service import (
    extract_indian_entities,
    structural_audit_strict,
    summarize_fallback,
    compute_legality_score as compute_legality_score_spec,
    derive_legal_status,
    sanitize_names,
    sanitize_orgs,
    transliterate_tamil_to_latin,
)
from backend.app.services.ml_service import classifier
from backend.app.core.deps import is_openai_ready


class IndianLegalVerifier:
    DOCUMENT_TYPE_KEYWORDS: Dict[str, List[str]] = {
        "Sale Deed": ["sale deed", "vendor", "purchaser", "consideration amount", "stamp duty", "sub-registrar", "schedule property"],
        "Lease Deed": ["lease deed", "lessor", "lessee", "rent", "lease period", "security deposit"],
        "Rental Agreement": ["rental agreement", "landlord", "tenant", "premises", "monthly rent"],
        "Gift Deed": ["gift deed", "donor", "donee", "gift", "love and affection"],
        "Patta": ["patta", "chitta", "adangal", "tahsildar", "survey number"],
        "Punjai Patta": ["punjai", "punjai patta", "dry land"],
        "TSLR": ["tslr", "town survey land register", "ward", "block number"],
        "Encumbrance Certificate": ["encumbrance certificate", "ec no", "sub-registrar", "encumbrance"],
        "Land Property": ["survey number", "taluk", "village", "fmb", "tslr", "layout approval", "dtcp", "cmda"],
        "Natham Certificate": ["natham", "natham certificate", "natham patta"],
        "Court Judgment": ["court judgment", "judgment", "order", "bench", "appellant", "respondent"],
        "Court Order": ["court order", "order passed", "hon'ble", "judge", "notary"],
        "Legal Notice": ["legal notice", "notice", "advocate", "reply within", "served upon"],
        "Affidavit": ["affidavit", "deponent", "solemnly affirm", "sworn before"],
        "Employment Contract": ["employment contract", "employer", "employee", "salary", "probation"],
        "Partnership Deed": ["partnership deed", "partners", "profit sharing", "firm"],
        "Corporate Agreement": ["agreement", "company", "authorized signatory", "governing law", "indemnity"],
        "Non Judicial Stamp Paper": ["non judicial stamp paper", "stamp paper", "bond paper", "non-judicial"],
        "PAN Card": ["pan", "income tax", "permanent account number"],
        "e-PAN": ["e-pan", "eaadhaar pan", "income tax department"],
        "Aadhaar Card": ["aadhaar", "uidai", "identity number"],
        "Aadhaar PVC Card": ["aadhaar pvc", "pvc card", "uidai"],
        "e-Aadhaar": ["e-aadhaar", "downloaded aadhaar", "uidai"],
        "Driving Licence": ["driving licence", "driving license", "dl no", "rto"],
        "Bank Statement": ["bank statement", "account number", "transaction", "debit", "credit"],
        "Invoice": ["invoice", "gst", "bill to", "amount due", "subtotal", "total"],
        "ID Card": ["identity card", "id card", "name", "issued by"],
    }

    STRUCTURE_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
        "Property": [
            ("stamp_duty", r"\bstamp duty\b"),
            ("registration_number", r"\b(registration|doc(?:ument)?)\s*no\.?\s*[:\-]?\s*[\w/-]+"),
            ("sub_registrar", r"\bsub[- ]registrar\b"),
            ("property_description", r"\b(schedule property|description of property|property schedule)\b"),
            ("signatures", r"\b(signature|signed by|executant|witness(?:es)?)\b"),
        ],
        "Court": [
            ("case_number", r"\b(c\.?s\.?|o\.?s\.?|case)\s*no\.?\s*[:\-]?\s*[\w/-]+"),
            ("court_name", r"\b(high court|district court|sessions court|supreme court)\b"),
            ("judge_or_notary", r"\b(judge|justice|notary)\b"),
            ("order_date", r"\b(order\s+dated|dated)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
            ("legal_sections", r"\b(section\s+\d+[A-Za-z]?\b|ipc|crpc|evidence act)\b"),
        ],
        "Business": [
            ("gstin", r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]"),
            ("company_registration", r"\b(CIN|company identification number)\b[:\-]?\s*[A-Z0-9]+"),
            ("authorized_signatory", r"\bauthorized signatory\b"),
            ("tax_breakdown", r"\b(tax|cgst|sgst|igst|cess)\b"),
        ],
        "Personal Certificates": [
            ("government_seal", r"\b(government of|govt\.? of|seal)\b"),
            ("certificate_number", r"\b(cert(?:ificate)?\s*no\.?|sl\.?\s*no\.?)\s*[:\-]?\s*[\w/-]+"),
            ("issuing_authority", r"\b(sub[- ]registrar|tahsildar|municipal corporation|district collector|authority)\b"),
            ("date_of_issue", r"\b(date of issue|issued on)\b[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),
        ],
        "Employment": [
            ("employer_details", r"\b(employer|company|organization)\b"),
            ("salary_structure", r"\b(salary|ctc|gross|net pay|components)\b"),
            ("hr_authorization", r"\b(hr|human resources|authorized)\b"),
        ],
    }

    DOC_TYPE_TO_CATEGORY: Dict[str, str] = {
        "Sale Deed": "Property",
        "Lease Deed": "Property",
        "Rental Agreement": "Property",
        "Gift Deed": "Property",
        "Patta": "Property",
        "Punjai Patta": "Property",
        "TSLR": "Property",
        "Encumbrance Certificate": "Property",
        "Land Property": "Property",
        "Natham Certificate": "Property",
        "Court Judgment": "Court",
        "Court Order": "Court",
        "Legal Notice": "Court",
        "Affidavit": "Court",
        "Employment Contract": "Employment",
        "Partnership Deed": "Business",
        "Corporate Agreement": "Business",
        "Non Judicial Stamp Paper": "Property",
        "PAN Card": "Personal Certificates",
        "e-PAN": "Personal Certificates",
        "Aadhaar Card": "Personal Certificates",
        "Aadhaar PVC Card": "Personal Certificates",
        "e-Aadhaar": "Personal Certificates",
        "Driving Licence": "Personal Certificates",
        "Bank Statement": "Business",
        "Invoice": "Business",
        "ID Card": "Personal Certificates",
    }

    def identify_document_type(self, text: str) -> Dict[str, Any]:
        t = (text or "")[:3000].lower()
        best_label = None
        best_hits = 0
        for label, anchors in self.DOCUMENT_TYPE_KEYWORDS.items():
            hits = sum(1 for a in anchors if a in t)
            if hits > best_hits:
                best_hits = hits
                best_label = label
        if best_hits >= 1:
            confidence = min(100, best_hits * 20)
            out = {
                "document_type": best_label,
                "confidence": confidence,
                "classification_source": "keyword",
                "fallback_reason": "",
            }
        else:
            try:
                res = classifier.predict_document_type(text or "")
                pred, conf = res[0], res[1]
                confidence = int(round(float(conf or 0.0) * 100))
                out = {
                    "document_type": pred,
                    "confidence": confidence,
                    "classification_source": "ml",
                    "fallback_reason": "",
                }
            except Exception as e:
                logging.exception(e)
                out = {
                    "document_type": "Other Indian Legal Document",
                    "confidence": 0,
                    "classification_source": "fallback",
                    "fallback_reason": "ml_error",
                }
        if int(out.get("confidence") or 0) < 40:
            out["document_type"] = "Other Indian Legal Document"
            out["classification_source"] = "fallback"
            out["fallback_reason"] = out.get("fallback_reason") or "low_confidence"
        return out

    def extract_key_information(self, text: str, doc_type: str) -> Dict[str, Any]:
        base = extract_indian_entities(text or "")
        reg_nums = []
        cert_nums = []
        issuing_auth = []
        addresses = []

        for m in re.finditer(r"(?:Doc(?:ument)?\s*No\.?|Reg(?:istration)?\s*No\.?)\s*[:\-]?\s*([\w/\-]+)", text or "", flags=re.IGNORECASE):
            reg_nums.append(m.group(1))
        for m in re.finditer(r"(?:Cert(?:ificate)?\s*No\.?|EC\s*No\.?|Sl\.?\s*No\.?)\s*[:\-]?\s*([\w/\-]+)", text or "", flags=re.IGNORECASE):
            cert_nums.append(m.group(1))
        for m in re.finditer(r"(?:Door|Flat|Plot|Survey|Village|District|State|Taluk|Mandal)\s*(?:No\.?)?\s*[:\-]?\s*[\w\s,/\-]+", text or "", flags=re.IGNORECASE):
            addresses.append(m.group(0).strip())
        for m in re.finditer(r"\b(Sub-Registrar|Tahsildar|Municipal Corporation|District Collector|Notary)\b", text or "", flags=re.IGNORECASE):
            issuing_auth.append(m.group(0).strip())
        # Tamil Revenue Village → add as organization label
        for m in re.finditer(r"வருவாய்\s*கிராமம்\s*[:\-]?\s*([^\n]+)", text or ""):
            val = transliterate_tamil_to_latin(m.group(1))
            if val:
                issuing_auth.append(f"Revenue Village {val}")

        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text or "")
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    issuing_auth.append(ent.text.strip())
        except Exception:
            pass

        t_upper = (text or "").upper()
        gstin_match = re.search(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]", t_upper)
        pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", t_upper)
        cin_match = re.search(r"\b[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b", t_upper)
        aadhaar_masked_match = re.search(r"\b[0-9X]{4}\s[0-9X]{4}\s[0-9]{4}\b", t_upper)

        id_numbers = {
            "gstin": gstin_match.group(0) if gstin_match else "Not Found",
            "pan": pan_match.group(0) if pan_match else "Not Found",
            "cin": cin_match.group(0) if cin_match else "Not Found",
            "aadhaar_masked": aadhaar_masked_match.group(0) if aadhaar_masked_match else "Not Found",
            "case_numbers": base.get("case_numbers") or [],
            "survey_numbers": base.get("survey_numbers") or [],
        }

        # Sanitize entities for verification UI
        base_sanitized = dict(base)
        try:
            base_sanitized["names"] = sanitize_names((base.get("parties") or []) + (base.get("role_parties") or []))
        except Exception:
            base_sanitized["names"] = base.get("parties") or []
        try:
            org_candidates = (issuing_auth or []) + (base.get("courts") or [])
            base_sanitized["organizations"] = sanitize_orgs(org_candidates)
        except Exception:
            base_sanitized["organizations"] = issuing_auth or []

        out = {
            "base_entities": base_sanitized,
            "registration_numbers": reg_nums or [],
            "certificate_numbers": cert_nums or [],
            "issuing_authority": issuing_auth or [],
            "address_details": addresses or [],
            "identification_numbers": id_numbers,
        }
        return out

    def validate_legal_structure(self, text: str, doc_type: str) -> Dict[str, Any]:
        category = self.DOC_TYPE_TO_CATEGORY.get(doc_type or "", "Other")
        templates = self.STRUCTURE_TEMPLATES.get(category, [])
        found = []
        missing = []
        for name, pattern in templates:
            if re.search(pattern, text or "", flags=re.IGNORECASE):
                found.append(name)
            else:
                missing.append(name)
        score = 0
        total = len(templates)
        if total > 0:
            score = int(round(100 * (len(found) / float(total))))
        try:
            audit = structural_audit_strict(text or "", doc_type or "")
        except Exception:
            audit = {}
        return {
            "category": category,
            "elements_found": found,
            "elements_missing": missing,
            "structure_score": score,
            "audit": audit,
        }

    def check_data_consistency(self, text: str, info: Dict[str, Any]) -> List[str]:
        issues = []
        ents = info.get("base_entities") or {}
        names = ents.get("names") or []
        orgs = ents.get("organizations") or []
        signers = ents.get("signers") or []
        parties = ents.get("parties") or []
        ids = info.get("identification_numbers") or {}
        gstin = ids.get("gstin") or "Not Found"
        pan = ids.get("pan") or "Not Found"
        cin = ids.get("cin") or "Not Found"
        aad = ids.get("aadhaar_masked") or "Not Found"
        if not names:
            issues.append("No names detected")
        if not signers:
            issues.append("No signatures detected")
        if gstin == "Not Found" and pan == "Not Found" and cin == "Not Found":
            if "invoice" in (text or "").lower():
                issues.append("Missing GSTIN/PAN on invoice")
        if aad != "Not Found" and len(aad.replace(" ", "")) != 12:
            issues.append("Aadhaar number format inconsistent")
        if len(set(orgs or []) & set(names or [])) > 0:
            issues.append("Organization name appears in person names")
        if parties and signers and len(set(parties) - set(signers)) > 0:
            issues.append("Parties differ from signers")
        regs = info.get("registration_numbers") or []
        if len(regs) > 1 and len(set(regs)) == 1:
            issues.append("Duplicate registration numbers across sections")
        return issues

    def detect_fraud_indicators(self, text: str, doc_type: str) -> List[str]:
        indicators = []
        try:
            detected, reason = classifier.detect_fraud(text or "")
            if detected:
                indicators.append(str(reason or "Fraud detected"))
        except Exception:
            pass
        try:
            an_detected, an_reason = classifier.detect_anomaly(text or "", doc_type or "Unknown", 0)
            if an_detected:
                indicators.append(str(an_reason or "Anomaly detected"))
        except Exception:
            pass
        lower = (text or "").lower()
        if "non judicial" in lower and "agreement" in lower and "witness" not in lower:
            indicators.append("Stamp paper without witness signatures")
        return indicators

    def compute_legality_score(self, struct: Dict[str, Any], info: Dict[str, Any]) -> float:
        analytics = {
            "signer_validation": {},
            "jurisdiction_curated": {},
            "clause_risk": {},
            "mandatory_fields": {
                "present": list(struct.get("elements_found") or []),
                "missing": list(struct.get("elements_missing") or []),
            },
        }
        try:
            return float(compute_legality_score_spec(analytics))
        except Exception:
            return 50.0

    def assign_risk_level(self, score: float, inconsistencies: List[str], fraud_indicators: List[str]) -> str:
        if fraud_indicators:
            return "HIGH"
        if score < 40 or len(inconsistencies) >= 3:
            return "HIGH"
        if score < 55 or len(inconsistencies) >= 1:
            return "MEDIUM"
        return "LOW"

    def generate_summary(self, document_type: str, legal_status: str, score: float, risk_level: str, text: str) -> str:
        base = f"Type: {document_type}. Status: {legal_status}. Legality score: {int(round(score))}/100. Risk: {risk_level}."
        try:
            if is_openai_ready():
                s = summarize_fallback(text or "")
                if s:
                    return base + " " + s
        except Exception:
            pass
        return base

    def verify(self, text: str) -> Dict[str, Any]:
        ident = self.identify_document_type(text or "")
        doc_type = ident.get("document_type") or "Unknown"
        info = self.extract_key_information(text or "", doc_type)
        struct = self.validate_legal_structure(text or "", doc_type)
        inconsistencies = self.check_data_consistency(text or "", info)
        fraud = self.detect_fraud_indicators(text or "", doc_type)
        score = self.compute_legality_score(struct, info)
        ml_ver = classifier.verify_document_ml(text or "")
        ai_conf = int(ml_ver.get("ai_confidence") or ident.get("confidence") or 0)
        marker = str(ml_ver.get("marker") or "").strip().title()
        analytics = {
            "document_type": doc_type,
            "ai_confidence": ai_conf,
            "legality_score": score,
            "verified_marker": marker,
            "mandatory_fields": {
                "present": list(struct.get("elements_found") or []),
                "missing": list(struct.get("elements_missing") or []),
            },
        }
        status_obj = derive_legal_status(analytics, ml_ver)
        legal_status = status_obj.get("status") or "UNKNOWN"
        summary_text = self.generate_summary(doc_type, legal_status, score, self.assign_risk_level(score, inconsistencies, fraud), text or "")
        ids = info.get("identification_numbers") or {}
        ids_norm = {
            "gstin": ids.get("gstin") or "Not Found",
            "pan": ids.get("pan") or "Not Found",
            "cin": ids.get("cin") or "Not Found",
            "aadhaar_masked": ids.get("aadhaar_masked") or "Not Found",
            "case_numbers": ids.get("case_numbers") or [],
            "survey_numbers": ids.get("survey_numbers") or [],
        }
        score_breakdown = {
            "structure_score": struct.get("structure_score") or 0,
            "ids_present": sum(1 for k in ("gstin", "pan", "cin", "aadhaar_masked") if (ids_norm.get(k) or "Not Found") != "Not Found"),
            "missing_elements_count": len(struct.get("elements_missing") or []),
            "fraud_flags_count": len(fraud or []),
        }
        actions = []
        if legal_status in {"UNKNOWN", "NEEDS_REVIEW"}:
            actions.append("Review document with a legal professional")
        if score < 55:
            actions.append("Add missing structural elements")
        if ids_norm.get("gstin") == "Not Found" and "invoice" in (text or "").lower():
            actions.append("Provide GSTIN for invoice validation")
        if ids_norm.get("pan") == "Not Found":
            actions.append("Include PAN where applicable")
        out = {
            "document_type": doc_type,
            "legal_status": legal_status,
            "summary": summary_text,
            "key_entities": info.get("base_entities") or {},
            "missing_elements": struct.get("elements_missing") or [],
            "inconsistencies_found": inconsistencies,
            "fraud_indicators": fraud,
            "risk_level": self.assign_risk_level(score, inconsistencies, fraud),
            "legality_score": score,
            "score_breakdown": score_breakdown,
            "recommended_actions": actions,
        }
        return out
