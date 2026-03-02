import re

def extract_entities(text):
    try:
        from backend.app.services.analysis_service import extract_indian_entities
        result = extract_indian_entities(text or "")
        return {
            "dates": result.get("dates", []),
            "amounts": result.get("money", []),
            "survey_numbers": result.get("survey_numbers", []),
            "case_numbers": result.get("case_numbers", []),
            "parties": result.get("parties", []),
            "role_parties": result.get("role_parties", []),
            "money": result.get("money", []),
            "registration_numbers": result.get("registration_numbers", []),
            "courts": result.get("courts", []),
            "land_types": result.get("land_types", []),
            "property_details": result.get("property_details", "")
        }
    except Exception:
        return {
            "dates": re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text or ""),
            "amounts": re.findall(r"₹\s?\d[\d,]*", text or ""),
            "survey_numbers": re.findall(r"(?i)S\.?No\.?\s?\d+/?\d*", text or ""),
            "case_numbers": re.findall(r"(?i)Case\sNo\.?\s?\d+", text or "")
        }
