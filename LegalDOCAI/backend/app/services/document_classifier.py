import re

def normalize_text(text):
    return (text or "").lower()

DOCUMENT_KEYWORDS = {
    "Sale Deed": ["sale deed", "vendor", "purchaser", "consideration"],
    "Rental Agreement": ["rental agreement", "landlord", "tenant", "rent"],
    "Gift Deed": ["gift deed", "donor", "donee"],
    "Power of Attorney": ["power of attorney", "attorney holder"],
    "Mortgage Document": ["mortgage", "loan against property"],
    "Patta": ["patta", "revenue department"],
    "TSLR": ["town survey land register", "tslr"],
    "Encumbrance Certificate": ["encumbrance certificate", "ec number"],
    "Court Judgment": ["judgment", "honourable court", "petitioner", "respondent"],
    "Court Order": ["order", "court order", "hereby ordered"],
    "Legal Notice": ["legal notice", "under instruction from my client"],
    "Affidavit": ["affidavit", "sworn", "deponent", "notary"],
    "Petition": ["petition", "prayer", "petitioner"],
    "Employment Contract": ["employment contract", "employer", "employee"],
    "Partnership Deed": ["partnership deed", "partners"],
    "MOA / AOA": ["memorandum of association", "articles of association"],
    "Aadhaar Card": ["aadhaar", "uidai"],
    "PAN Card": ["income tax department", "permanent account number"],
    "Voter ID": ["election commission", "voter id"],
    "Driving Licence": ["driving licence", "dl no"],
    "Passport": ["passport", "republic of india"],
    "Birth Certificate": ["birth certificate", "date of birth"],
    "Income Certificate": ["income certificate"],
    "Caste Certificate": ["caste certificate"],
    "Domicile Certificate": ["domicile certificate"],
    "Insurance Document": ["insurance policy", "policy number"],
    "Bank Statement": ["account statement", "bank"]
}

CATEGORY_MAP = {
    "Sale Deed": "Land/Legal Document",
    "Rental Agreement": "Legal Document",
    "Gift Deed": "Legal Document",
    "Power of Attorney": "Legal Document",
    "Mortgage Document": "Legal Document",
    "Patta": "Land Document",
    "TSLR": "Land Document",
    "Encumbrance Certificate": "Land Document",
    "Court Judgment": "Legal Document",
    "Court Order": "Legal Document",
    "Legal Notice": "Legal Document",
    "Affidavit": "Legal Document",
    "Petition": "Legal Document",
    "Employment Contract": "Employment Document",
    "Partnership Deed": "Legal Document",
    "MOA / AOA": "Legal Document",
    "Aadhaar Card": "Identity Document",
    "PAN Card": "Identity Document",
    "Voter ID": "Identity Document",
    "Driving Licence": "Identity Document",
    "Passport": "Identity Document",
    "Birth Certificate": "Certificate",
    "Income Certificate": "Certificate",
    "Caste Certificate": "Certificate",
    "Domicile Certificate": "Certificate",
    "Insurance Document": "Financial Document",
    "Bank Statement": "Financial Document"
}

def classify_document(text):
    t = normalize_text(text)
    for doc_type, keywords in DOCUMENT_KEYWORDS.items():
        for word in keywords:
            if word in t:
                category = CATEGORY_MAP.get(doc_type, "Other")
                return {"category": category, "document_type": doc_type}
    return {"category": "Other", "document_type": "Other"}
