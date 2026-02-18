import re

def extract_entities(text):
    return {
        "dates": re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text or ""),
        "amounts": re.findall(r"₹\s?\d[\d,]*", text or ""),
        "survey_numbers": re.findall(r"(?i)S\.?No\.?\s?\d+/?\d*", text or ""),
        "case_numbers": re.findall(r"(?i)Case\sNo\.?\s?\d+", text or "")
    }
