import os
import joblib
import numpy as np
import re
import json
import csv
import hashlib
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple, Iterable, Optional, Callable

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

# Constants
MODEL_VERSION = "2.1.0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert_classifier")
MODEL_PATH = os.path.join(MODEL_DIR, "doc_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ALT_MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
ALT_PIPELINE_PATH = os.path.join(ALT_MODELS_DIR, "document_model.pkl")
STAGE1_PIPELINE_PATH = os.path.join(ALT_MODELS_DIR, "document_stage1_model.pkl")
AUTO_TRAIN_PATH = os.path.join(BASE_DIR, "legal_dataset", "auto_training_data.json")
AUTO_TRAIN_STATS_PATH = os.path.join(BASE_DIR, "legal_dataset", "auto_training_stats.json")

LEGAL_CATEGORIES = [
    "Sale Deed",
    "Lease Deed",
    "Rental Agreement",
    "Gift Deed",
    "Patta",
    "Punjai Patta",
    "TSLR",
    "Encumbrance Certificate",
    "Land Property",
    "Natham Certificate",
    "Court Judgment",
    "Court Order",
    "Legal Notice",
    "Affidavit",
    "Employment Contract",
    "Partnership Deed",
    "Corporate Agreement",
    "Non Judicial Stamp Paper",
    "Other Indian Legal Document",
]

NON_LEGAL_CATEGORIES = [
    "Resume",
    "Invoice",
    "ID Card",
    "Driving Licence",
    "Bank Statement",
    "Marksheet",
    "Admit Card",
    "Transfer Certificate",
    "Bonafide Certificate",
    "Application Interface Screenshot",
    "Other Non Legal Document",
]

ALL_SUPPORTED_CATEGORIES = LEGAL_CATEGORIES + NON_LEGAL_CATEGORIES
GENERIC_RULE_LABELS = {"Legal Document", "Legal Agreement", "Contract"}

CATEGORY_ALIASES = {
    "legal agreement": "Corporate Agreement",
    "contract": "Corporate Agreement",
    "company agreement": "Corporate Agreement",
    "moa / aoa": "Corporate Agreement",
    "regulatory filing": "Other Indian Legal Document",
    "petition": "Other Indian Legal Document",
    "chitta": "Other Indian Legal Document",
    "settlement deed": "Other Indian Legal Document",
    "partition deed": "Other Indian Legal Document",
    "release deed": "Other Indian Legal Document",
    "rectification deed": "Other Indian Legal Document",
    "fmb sketch": "Other Indian Legal Document",
    "layout approval": "Other Indian Legal Document",
    "native certificate": "Other Indian Legal Document",
    "insurance document": "Other Indian Legal Document",
    "will": "Other Indian Legal Document",
    "power of attorney": "Other Indian Legal Document",
    "mortgage deed": "Other Indian Legal Document",
    "bond paper": "Non Judicial Stamp Paper",
    "bond": "Non Judicial Stamp Paper",
    "stamp paper": "Non Judicial Stamp Paper",
    "non judicial stamp paper": "Non Judicial Stamp Paper",
    "non-judicial stamp paper": "Non Judicial Stamp Paper",
    "employment": "Employment Contract",
    "encumbrance": "Encumbrance Certificate",
    "town survey land register (tslr)": "TSLR",
    "natham": "Natham Certificate",
    "natham patta": "Natham Certificate",
    "natham certificate": "Natham Certificate",
    "natham settlement": "Natham Certificate",
    "punjai": "Punjai Patta",
    "punja": "Punjai Patta",
    "punjai patta": "Punjai Patta",
    "dry land patta": "Punjai Patta",
    "aadhaar card": "Aadhaar Card",
    "aadhaar pvc card": "Aadhaar PVC Card",
    "e-aadhaar": "e-Aadhaar",
    "eaadhaar": "e-Aadhaar",
    "e aadhaar": "e-Aadhaar",
    "driving license": "Driving Licence",
    "driver license": "Driving Licence",
    "driving licence": "Driving Licence",
    "driver licence": "Driving Licence",
    "dl": "Driving Licence",
    "licence document": "Driving Licence",
    "license document": "Driving Licence",
    "cv": "Resume",
    "curriculum vitae": "Resume",
    "driving licence card": "Driving Licence",
    "drivers license": "Driving Licence",
    "driver's license": "Driving Licence",
    "voter id": "Voter ID",
    "voter identity card": "Voter ID",
    "e-epic": "e-Voter ID",
    "epic": "Voter ID",
    "ration card": "Ration Card",
    "e-ration card": "e-Ration Card",
    "pan card": "PAN Card",
    "e-pan": "e-PAN",
    "passport": "Passport",
    "oci card": "OCI Card",
    "pio card": "PIO Card",
    "id proof": "ID Card",
    "other document": "Other Non Legal Document",
    "unknown": "Other Non Legal Document",
    "application interface screenshot": "Application Interface Screenshot",
    "app interface screenshot": "Application Interface Screenshot",
    "ui screenshot": "Application Interface Screenshot",
    "website screenshot": "Application Interface Screenshot",
    "generic legal document": "Other Indian Legal Document",
    "generic legal document": "Other Indian Legal Document",
    "marksheet": "Marksheet",
    "mark sheet": "Marksheet",
    "grade sheet": "Marksheet",
    "score card": "Marksheet",
    "admit card": "Admit Card",
    "hall ticket": "Admit Card",
    "transfer certificate": "Transfer Certificate",
    "tc": "Transfer Certificate",
    "bonafide certificate": "Bonafide Certificate",
    "bonafide": "Bonafide Certificate",
    "land property": "Land Property",
}

NON_LEGAL_HINTS = {
    "Resume": [
        "resume",
        "curriculum vitae",
        "objective",
        "professional summary",
        "education",
        "experience",
        "skills",
        "projects",
        "internship",
        "cgpa",
        "diploma",
    ],
    "Invoice": ["invoice", "tax invoice", "gst", "amount due", "bill to", "subtotal", "total"],
    "ID Card": ["aadhaar", "passport", "pan", "identity card"],
    "Driving Licence": ["driving licence", "driving license", "dl no", "dl", "rto"],
    # Broaden licence detection for sparse OCR
    "Driving Licence": ["driving licence", "driving license", "licence", "license", "dl no", "dl", "rto"],
    "Bank Statement": ["bank statement", "account number", "transaction", "debit", "credit", "ifsc"],
    "Marksheet": ["marksheet", "mark sheet", "grade sheet", "score card", "result", "examination", "board of secondary education", "university"],
    "Admit Card": ["admit card", "hall ticket", "examination", "candidate", "roll no", "exam"],
    "Transfer Certificate": ["transfer certificate", "tc", "school", "college"],
    "Bonafide Certificate": ["bonafide certificate", "bonafide", "student", "college", "institute"],
    "Application Interface Screenshot": [
        "overview", "risks", "entities", "timeline", "chat", "comparison", "report",
        "ask about your document", "ask", "answer:", "analysis", "upload", "document history"
    ],
}

CATEGORY_DESCRIPTIONS = {
    "Sale Deed": "A legal property transfer deed between seller and buyer with consideration and schedule property.",
    "Lease Deed": "A legal lease deed with lessor, lessee, lease term, rent and possession conditions.",
    "Rental Agreement": "A rental contract with tenant, landlord, monthly rent, deposit and duration terms.",
    "Gift Deed": "A deed where donor transfers property to donee without consideration.",
    "Patta": "Tamil Nadu land ownership/revenue record with survey number and tahsildar references.",
    "Punjai Patta": "Dry-land (punjai) land record or patta for rural/agricultural property.",
    "TSLR": "Town Survey Land Register record containing ward, block and town survey details.",
    "Encumbrance Certificate": "Sub-registrar certificate showing encumbrance transactions and registration history.",
    "Natham Certificate": "Village natham land record/certificate issued by local authorities.",
    "Court Judgment": "Court judgment text with petitioner, respondent, findings and final decision.",
    "Court Order": "Court order with directions, interim/final order language and compliance instructions.",
    "Legal Notice": "Advocate legal notice invoking sections, demand, breach and response timeline.",
    "Affidavit": "Sworn affidavit statement signed before notary/deponent declaration.",
    "Employment Contract": "Employment agreement between employer and employee with salary and terms.",
    "Partnership Deed": "Business partnership deed with partners, share ratios and firm clauses.",
    "Corporate Agreement": "Corporate/commercial agreement such as MSA, shareholders or board-level contract.",
    "Non Judicial Stamp Paper": "Non-judicial stamp paper used to execute legal documents and agreements.",
    "Other Indian Legal Document": "Indian legal/government record not matching main legal categories.",
    "Resume": "Candidate profile with education, experience, skills and projects.",
    "Invoice": "Billing invoice with totals, taxes, invoice number and amount due.",
    "ID Card": "Identity card document such as Aadhaar, PAN, passport or voter ID.",
    "Driving Licence": "Government-issued driving licence (DL) with holder details and validity.",
    "Bank Statement": "Bank account statement listing transactions, balances and account details.",
    "Other Non Legal Document": "General non-legal document that does not match known non-legal categories.",
}

def _resume_like_score(text: str) -> int:
    t = (text or "").lower()
    if not t:
        return 0
    cues = [
        "resume",
        "curriculum vitae",
        "objective",
        "professional summary",
        "education",
        "experience",
        "skills",
        "projects",
        "internship",
        "cgpa",
        "diploma",
    ]
    return sum(1 for c in cues if c in t)

REQUIRED_KEYWORDS = {
    "Sale Deed": ["vendor", "purchaser", "consideration"],
    "Lease Deed": ["lessor", "lessee", "lease term"],
    "Rental Agreement": ["tenant", "rent", "landlord"],
    "Patta": ["tahsildar", "survey number"],
    "TSLR": ["town survey", "ward", "block"],
    "Encumbrance Certificate": ["sub-registrar", "encumbrance"],
    "Encumbrance": ["sub-registrar", "encumbrance"],
    "Court Judgment": ["judgment", "honorable court"],
    "Court Order": ["order", "interim", "directed"],
    "Legal Notice": ["section", "notice", "payment"],
    "Affidavit": ["sworn", "notary"],
    "Employment Contract": ["employee", "employer", "salary"],
    "Employment": ["employee", "employer", "salary"],
    "Non Judicial Stamp Paper": ["non judicial", "stamp paper"],
    "Natham Certificate": ["natham"],
    "Punjai Patta": ["punjai"],
}

DOC_TYPE_RULES = [
    {"label": "Non Judicial Stamp Paper", "all": [], "any": ["non judicial", "non-judicial", "stamp paper", "bond paper", "bond", "impressed stamp", "india non judicial", "non judicial stamp"], "regex": [r"\bnon[-\s]?judicial\b", r"\bstamp\s+paper\b", r"\bbond\b"]},
    {"label": "Natham Certificate", "all": [], "any": ["natham", "natham patta", "natham settlement", "village natham", "natham land"], "regex": [r"\bnatham\b"]},
    {"label": "Punjai Patta", "all": [], "any": ["punjai", "punja", "punjai patta", "dry land patta", "dry land"], "regex": [r"\bpunjai\b", r"\bpunja\b"]},
    {"label": "Patta", "all": [], "any": ["patta", "பட்டா", "land patta", "patta passbook", "ryotwari", "eservices.tn.gov.in", "tn.gov.in"], "regex": [r"\bsurvey\s*(?:no|number)\b", r"சர்வே\s*எண்", r"\btahsildar\b", r"(தாசில்தார்|தாலுகா)", r"(கிராமம்|வட்டம்)"]},
    {"label": "Chitta", "all": [], "any": ["chitta", "a-register", "adangal", "village account"]},
    {"label": "Encumbrance Certificate", "all": [], "any": ["encumbrance certificate", "ec", "sub-registrar", "search period", "nil encumbrance"], "regex": [r"\bencumbrance\b", r"\bec\b"]},
    {"label": "TSLR", "all": [], "any": ["tslr", "town survey land register", "town survey", "ts no", "ward", "block"]},
    {"label": "FMB Sketch", "all": [], "any": ["fmb", "field measurement book", "sub division", "survey sketch", "field sketch"]},
    {"label": "Layout Approval", "all": [], "any": ["layout approval", "dtcp", "cmda", "approved layout", "planning permit"]},
    {"label": "Sale Deed", "all": [], "any": ["sale deed", "deed of sale", "absolute sale", "vendor", "vendee", "schedule property", "consideration", "conveyance", "purchaser", "seller"]},
    {"label": "Settlement Deed", "all": [], "any": ["settlement deed", "settlor", "beneficiary", "family settlement"]},
    {"label": "Partition Deed", "all": [], "any": ["partition deed", "partition", "co-parceners", "coparceners", "share allotment"]},
    {"label": "Release Deed", "all": [], "any": ["release deed", "relinquishment deed", "release", "relinquish"]},
    {"label": "Rectification Deed", "all": [], "any": ["rectification deed", "correction deed", "rectify", "typographical error"]},
    {"label": "Land Property", "all": [], "any": ["patta","பட்டா","tslr","fmb","town survey","field measurement book","layout approval","dtcp","cmda","survey number","survey no","சர்வே","tahsildar","தாசில்தார்","taluk","தாலுகா","village","கிராமம்","property","plot","acre","hectare","ward","block","registration","sub registrar","schedule property","eservices.tn.gov.in","tn.gov.in"], "regex": [r"\bsurvey\s*(?:no|number)\b", r"சர்வே\s*(?:எண்|number)", r"\btahsildar\b", r"(தாசில்தார்|தாலுகா)", r"\btslr\b", r"\bfmb\b", r"\bsub[-\s]?registrar\b"]},
    {"label": "Rental Agreement", "all": [], "any": ["rental agreement", "lease agreement", "lease deed", "landlord", "tenant", "lessor", "lessee", "monthly rent", "security deposit", "premises"]},
    {"label": "Resume", "all": [], "any": ["resume", "curriculum vitae", "cv", "education", "experience", "skills", "projects"], "regex": [r"\bcurriculum vitae\b", r"\bresume\b"]},
    {"label": "Invoice", "all": [], "any": ["invoice", "tax invoice", "bill", "amount due", "gst", "invoice no", "invoice number"], "regex": [r"\binvoice\b", r"\bamount due\b", r"\bgst\b"]},
    {"label": "ID Card", "all": [], "any": ["id card", "identity card", "passport", "driver licence", "driver license", "pan card"], "regex": [r"\bpassport\b", r"\bpan\b"]},
    {"label": "Bank Statement", "all": [], "any": ["bank statement", "account number", "transaction", "debit", "credit", "balance", "ifsc"], "regex": [r"\bbank statement\b", r"\baccount number\b", r"\btransaction\b"]},
    {"label": "Legal Notice", "all": [], "any": ["legal notice", "notice", "advocate", "demand notice"], "regex": [r"\blegal notice\b", r"\bdemand notice\b"]},
    {"label": "Lease Deed", "all": [], "any": ["lease deed", "lessor", "lessee", "rent", "lease term"]},
    {"label": "Power of Attorney", "all": [], "any": ["power of attorney", "principal", "agent", "poa", "constituted attorney"]},
    {"label": "Mortgage Deed", "all": [], "any": ["mortgage deed", "mortgagor", "mortgagee", "equitable mortgage", "loan secured", "hypothecation"]},
    {"label": "Affidavit", "all": ["affidavit"], "any": ["deponent", "solemnly", "affirm", "sworn"]},
    {"label": "Employment Contract", "all": [], "any": ["employment contract", "employee", "employer", "salary", "position", "probation", "notice period"]},
    {"label": "Partnership Deed", "all": [], "any": ["partnership deed", "partners", "firm", "profit sharing", "partnership"]},
    {"label": "Corporate Agreement", "all": [], "any": ["corporate agreement", "master service agreement", "msa", "shareholders agreement", "shareholders", "nda", "service level agreement", "sla", "memorandum of understanding", "mou", "vendor agreement"]},
    {"label": "MOA / AOA", "all": [], "any": ["memorandum of association", "moa", "articles of association", "aoa", "company incorporation"]},
    {"label": "Regulatory Filing", "all": [], "any": ["regulatory filing", "roc filing", "mca", "form 32", "form 8", "form 23", "form mgt-7", "form aoc-4", "cin", "din"], "regex": [r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", r"\bDIN\s*[:\-]?\s*\d{8}\b"]},
    {"label": "Native Certificate", "all": ["native", "certificate"], "any": ["nativity"]},
    {"label": "Insurance Document", "all": ["insurance"], "any": ["policy", "premium", "insurer", "insured"]},
    {"label": "Gift Deed", "all": [], "any": ["gift deed", "deed of gift", "donor", "donee"]},
    {"label": "Will", "all": ["will"], "any": ["testator", "bequeath", "legacy", "testament"]},
    {"label": "Court Order", "all": [], "any": ["court order", "ordered that", "it is hereby ordered", "interim order", "final order"]},
    {"label": "Court Judgment", "all": [], "any": ["judgment", "judgement", "petitioner", "respondent", "pronounced on", "high court", "supreme court"]},
    {"label": "Legal Notice", "all": [], "any": ["legal notice", "hereby called upon", "notice under", "cause of action", "failing which"]},
    {"label": "Petition", "all": [], "any": ["petition", "writ petition", "civil petition", "criminal petition", "petitioner", "respondent"]},
    {"label": "e-Aadhaar", "all": ["aadhaar"], "any": ["e-aadhaar", "eaadhaar", "e aadhaar", "unique identification authority"]},
    {"label": "Aadhaar PVC Card", "all": ["aadhaar", "pvc"], "any": []},
    {"label": "Aadhaar Card", "all": ["aadhaar"], "any": []},
    {"label": "e-PAN", "all": [], "any": ["e-pan", "epan", "income tax department", "govt. of india", "govt of india"], "regex": [r"\b[A-Z]{5}\d{4}[A-Z]\b"]},
    {"label": "PAN Card", "all": [], "any": ["pan", "income tax", "income tax department", "permanent account number", "govt. of india", "govt of india", "date of birth"], "regex": [r"\b[A-Z]{5}\d{4}[A-Z]\b"]},
    {"label": "e-Voter ID", "all": ["voter"], "any": ["e-epic", "e voter", "e-voter", "epic"]},
    {"label": "Voter ID", "all": ["voter"], "any": ["epic", "election commission"], "regex": [r"\b[A-Z]{3}\d{7}\b"]},
    {"label": "e-Driving Licence", "all": [], "any": ["e-driving", "e driving", "digilocker", "driving licence", "driving license", "ministry of road transport and highways", "transport department"]},
    {"label": "Driving Licence", "all": [], "any": ["driving", "licence", "license", "rto", "dl no", "ministry of road transport and highways", "transport department"], "regex": [r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", r"\bDL[-\s]?\d{10,16}\b"]},
    {"label": "Learner Licence", "all": ["learner"], "any": ["licence", "license", "ll no", "learning"]},
    {"label": "Vehicle RC", "all": ["registration", "certificate"], "any": ["rc", "chassis", "vehicle"], "regex": [r"\b[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}\b", r"\b(chassis|chassis no\.?|vin)\s*[:\-]?\s*[A-HJ-NPR-Z0-9]{17}\b"]},
    {"label": "PUC Certificate", "all": ["pollution", "certificate"], "any": ["puc"]},
    {"label": "Motor Insurance", "all": ["insurance"], "any": ["motor", "vehicle", "policy"], "regex": [r"\bpolicy\s*no\.?\s*[:\-]?\s*[A-Z0-9\-]{8,20}\b"]},
    {"label": "Passport", "all": ["passport"], "any": ["nationality", "republic of india"], "regex": [r"\b[A-Z]\d{7}\b", r"\bP<[A-Z]{3}[A-Z<]{39}\b", r"\b[0-9]{7}[A-Z0-9<]{1}\b"]},
    {"label": "OCI Card", "all": ["oci"], "any": ["overseas citizen"]},
    {"label": "PIO Card", "all": ["pio"], "any": ["person of indian origin"]},
    {"label": "Birth Certificate", "all": ["birth", "certificate"], "any": ["date of birth"]},
    {"label": "Death Certificate", "all": ["death", "certificate"], "any": ["date of death"]},
    {"label": "Marriage Certificate", "all": ["marriage", "certificate"], "any": ["registration"]},
    {"label": "Divorce Decree", "all": ["divorce"], "any": ["decree", "family court"]},
    {"label": "Adoption Certificate", "all": ["adoption", "certificate"], "any": ["adoptive"]},
    {"label": "Caste Certificate", "all": ["caste", "certificate"], "any": []},
    {"label": "Income Certificate", "all": ["income", "certificate"], "any": ["annual", "revenue"]},
    {"label": "Domicile Certificate", "all": ["domicile", "certificate"], "any": []},
    {"label": "Residence Certificate", "all": ["residence", "certificate"], "any": ["address"]},
    {"label": "EWS Certificate", "all": ["ews", "certificate"], "any": ["economically weaker"]},
    {"label": "Non-Creamy Layer Certificate", "all": ["non", "creamy", "layer"], "any": ["obc"]},
    {"label": "Disability Certificate", "all": ["disability", "certificate"], "any": ["percentage"]},
    {"label": "Senior Citizen Certificate", "all": ["senior", "citizen"], "any": ["age"]},
    {"label": "Ration Card", "all": ["ration", "card"], "any": ["fps", "food"]},
    {"label": "e-Ration Card", "all": ["ration", "card"], "any": ["e-ration", "digital"]},
    {"label": "Legal Heir Certificate", "all": ["legal", "heir"], "any": ["deceased"]},
    {"label": "Transfer Certificate", "all": ["transfer", "certificate"], "any": ["school", "tc"]},
    {"label": "Bonafide Certificate", "all": ["bonafide", "certificate"], "any": ["student", "institution"]},
    {"label": "Migration Certificate", "all": ["migration", "certificate"], "any": ["board", "university"]},
    {"label": "Employee ID", "all": ["employee", "id"], "any": ["company", "department"]},
    {"label": "Experience Certificate", "all": ["experience", "certificate"], "any": ["employment"]},
    {"label": "Salary Certificate", "all": ["salary", "certificate"], "any": ["employer"]},
    {"label": "Labour Card", "all": ["labour", "card"], "any": ["worker"]},
    {"label": "NREGA Job Card", "all": ["nrega", "job", "card"], "any": ["mgnrega"]},
    {"label": "ESIC Card", "all": ["esic", "card"], "any": ["insurance"], "regex": [r"\b\d{10}\b"]},
    {"label": "GST Certificate", "all": ["gst", "certificate"], "any": ["gstin"], "regex": [r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"]},
    {"label": "Trade License", "all": ["trade", "license"], "any": ["municipal"]},
    {"label": "Professional Tax Certificate", "all": ["professional", "tax"], "any": ["pt"]},
    {"label": "Farmer ID", "all": ["farmer", "id"], "any": ["agriculture"]},
    {"label": "Kisan Credit Card", "all": ["kisan", "credit", "card"], "any": ["kcc"]},
    {"label": "Soil Health Card", "all": ["soil", "health", "card"], "any": ["test"]},
    {"label": "Ayushman Card", "all": ["ayushman", "card"], "any": ["pm-jay", "beneficiary"]},
    {"label": "Medical Fitness Certificate", "all": ["medical", "fitness", "certificate"], "any": ["doctor"]},
    {"label": "Bank Passbook", "all": ["passbook"], "any": ["account", "ifsc"], "regex": [r"\b[A-Z]{4}0[A-Z0-9]{6}\b"]},
    {"label": "Bank Statement", "all": ["bank", "statement"], "any": ["transaction"], "regex": [r"\b[A-Z]{4}0[A-Z0-9]{6}\b"]},
    {"label": "Cancelled Cheque", "all": ["cheque"], "any": ["cancelled", "canceled"]},
    {"label": "Debit Card", "all": ["debit", "card"], "any": ["bank"]},
    {"label": "Credit Card Statement", "all": ["credit", "card", "statement"], "any": ["outstanding"]},
    # --- General document types (added for broader coverage) ---
    {"label": "Resume", "all": [], "any": ["resume", "curriculum vitae", "cv", "education", "experience", "skills"], "regex": [r"\b(Education|Experience|Skills|Projects|Summary)\b"]},
    {"label": "Invoice", "all": [], "any": ["invoice", "tax invoice", "gst", "gstin", "bill to", "invoice number", "subtotal", "total", "amount due"], "regex": [r"\b(invoice\s*(no|number)\s*[:\-]?\s*[A-Za-z0-9\-]+)\b", r"\bGSTIN\b"]},
    {"label": "ID Card", "all": [], "any": ["identity card", "id card", "id no", "date of birth", "govt of", "government of"], "regex": [r"(?i)\b(id\s*(no|number)\s*[:\-]?\s*[A-Za-z0-9\-]+)\b"]},
    {"label": "Contract", "all": [], "any": ["contract", "contract agreement"], "regex": [r"\b(contract)\b"]},
    {"label": "Affidavit", "all": [], "any": ["affidavit", "sworn statement", "deponent", "notary"], "regex": [r"\baffidavit\b"]},
    {"label": "Legal Agreement", "all": [], "any": ["agreement", "between", "parties", "witnesseth"], "regex": [r"\bagreement\b", r"\bbetween\b"]},
    {"label": "Legal Document", "all": [], "any": ["hereby", "witnesseth", "whereas", "parties", "agreement", "deed", "dated", "between", "signed", "executed"]},
]

_DOC_TYPE_RULES_COMPILED = []
for r in DOC_TYPE_RULES:
    compiled = []
    for pat in r.get("regex", []):
        try:
            compiled.append(re.compile(pat))
        except Exception:
            continue
    _DOC_TYPE_RULES_COMPILED.append({**r, "regex_compiled": compiled})

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.stage1_pipeline = None
        self.is_loaded = False
        self._pred_cache = {}
        self._semantic_model = None
        self._semantic_label_embeddings = None
        
        # BERT primary classifier state
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_device = "cuda" if torch_available and torch.cuda.is_available() else "cpu"
        
        self._load_model()
        self._load_bert_model()

    def _load_bert_model(self):
        """
        Loads the primary BERT classifier from disk.
        """
        if os.environ.get("DISABLE_BERT") == "1":
            return
            
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if os.path.exists(BERT_MODEL_DIR):
                print(f"[DocumentClassifier] Loading BERT model from {BERT_MODEL_DIR}...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
                self.bert_model.to(self.bert_device)
                self.bert_model.eval()
        except Exception as e:
            print(f"[DocumentClassifier] BERT loading failed: {e}")

    def _predict_bert_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Batch BERT inference logic for improved performance.
        """
        if not self.bert_model or not self.bert_tokenizer:
            return [(self.predict_document_type(t)[:2]) for t in texts]
            
        try:
            import torch
            inputs = self.bert_tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            ).to(self.bert_device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confs, idxs = torch.max(probs, dim=-1)
                
            results = []
            for i in range(len(texts)):
                label = self.bert_model.config.id2label[idxs[i].item()]
                results.append((label, confs[i].item()))
            return results
        except Exception as e:
            print(f"[DocumentClassifier] BERT batch prediction failed: {e}")
            return [self.predict_document_type(t)[:2] for t in texts]

    def predict_with_sliding_window(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[str, float]:
        """
        Implementation of chunk voting strategy for long documents.
        Processes text in windows and aggregates votes for the final category.
        """
        if not text:
            return "Other Non Legal Document", 0.0
            
        # 1. Chunking
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) > 100: # Ignore tiny fragments
                chunks.append(chunk)
            if len(chunks) >= 8: # Cap at 8 chunks for performance
                break
                
        if not chunks:
            res = self.predict_document_type(text)
            return res[0], res[1]
            
        # 2. Collect votes using Batch BERT if available
        votes = {}
        
        # Determine if we should use BERT for chunks
        if self.bert_model:
            results = self._predict_bert_batch(chunks)
        else:
            results = [self.predict_document_type(c)[:2] for c in chunks]

        for label, conf in results:
            label = self.canonicalize_category(label)
            if label not in votes:
                votes[label] = []
            votes[label].append(conf)
            
        # 3. Aggregate results
        final_scores = {label: (len(confs) * (sum(confs) / len(confs))) for label, confs in votes.items()}
        best_label = max(final_scores, key=final_scores.get)
        final_conf = sum(votes[best_label]) / len(votes[best_label])
        
        return best_label, final_conf

    def _predict_bert(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Primary BERT-based inference logic.
        """
        if not self.bert_model or not self.bert_tokenizer:
            return None
            
        try:
            import torch
            inputs = self.bert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            ).to(self.bert_device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                conf, idx = torch.max(probs, dim=-1)
                
            label = self.bert_model.config.id2label[idx.item()]
            return label, conf.item()
        except Exception as e:
            print(f"[DocumentClassifier] BERT prediction failed: {e}")
            return None

    @staticmethod
    def canonicalize_category(label: Optional[str]) -> str:
        raw = (label or "").strip()
        if not raw:
            return "Other Non Legal Document"
        low = raw.lower()
        if raw in ALL_SUPPORTED_CATEGORIES:
            return raw
        if low in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[low]
        return "Other Indian Legal Document" if ("deed" in low or "court" in low or "legal" in low) else "Other Non Legal Document"

    @staticmethod
    def is_legal_category(label: Optional[str]) -> bool:
        return DocumentClassifier.canonicalize_category(label) in LEGAL_CATEGORIES

    def _load_model(self):
        try:
            if os.path.exists(ALT_PIPELINE_PATH):
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                self.pipeline = joblib.load(ALT_PIPELINE_PATH)
                self.model = getattr(self.pipeline, "named_steps", {}).get("clf", None)
                self.vectorizer = getattr(self.pipeline, "named_steps", {}).get("tfidf", None)
                self.is_loaded = self._validate_loaded_model()
                if self.is_loaded:
                    print(f"ML Pipeline loaded from {ALT_PIPELINE_PATH}")
                else:
                    print(f"ML Pipeline at {ALT_PIPELINE_PATH} is incompatible/unfitted. Will fallback.")
            elif os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.is_loaded = self._validate_loaded_model()
                if self.is_loaded:
                    print(f"ML Model loaded from {MODEL_PATH}")
                else:
                    print(f"ML Model at {MODEL_PATH} is incompatible/unfitted. Will fallback.")
            else:
                print(f"ML Model not found at {MODEL_PATH}")
            if os.path.exists(STAGE1_PIPELINE_PATH):
                try:
                    self.stage1_pipeline = joblib.load(STAGE1_PIPELINE_PATH)
                except Exception as e:
                    self.stage1_pipeline = None
                    print(f"Failed to load Stage1 model: {e}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")

    def _validate_loaded_model(self) -> bool:
        try:
            sample = "This legal agreement between parties includes liability and termination clauses."
            if self.pipeline is not None:
                _ = self.pipeline.predict([sample])[0]
                return True
            if self.model is None or self.vectorizer is None:
                return False
            X = self.vectorizer.transform([sample])
            _ = self.model.predict(X)[0]
            return True
        except Exception:
            self.pipeline = None
            self.model = None
            self.vectorizer = None
            return False

    def get_supported_document_types(self) -> List[str]:
        return list(ALL_SUPPORTED_CATEGORIES)

    def train_model(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        try:
            if not texts or not labels:
                return {"success": False, "error": "No training data provided"}

            pairs = []
            for t, l in zip(texts, labels):
                txt = (t or "").strip()
                if not txt:
                    continue
                pairs.append((txt, self.canonicalize_category(l)))
            if not pairs:
                return {"success": False, "error": "No valid training samples after normalization"}

            train_texts = [p[0] for p in pairs]
            train_labels = [p[1] for p in pairs]
            stage1_labels = ["Legal" if self.is_legal_category(lbl) else "NonLegal" for lbl in train_labels]

            print(f"Training model with {len(train_texts)} normalized samples...")
            # Stage-2 pipeline: category classification
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=25000,
                min_df=2,
                max_df=0.9,
                sublinear_tf=True,
                stop_words="english",
            )
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
            )
            pipe = Pipeline([("tfidf", self.vectorizer), ("clf", clf)])
            pipe.fit(train_texts, train_labels)
            # Extract fitted steps for backward compatibility
            self.pipeline = pipe
            self.model = pipe.named_steps.get("clf")
            self.vectorizer = pipe.named_steps.get("tfidf")

            # Stage-1 binary legal/non-legal classifier
            stage1_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=2, max_df=0.95, stop_words="english")),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ])
            stage1_pipe.fit(train_texts, stage1_labels)
            self.stage1_pipeline = stage1_pipe

            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(ALT_MODELS_DIR, exist_ok=True)

            # Save both standalone artifacts and full pipeline
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            joblib.dump(pipe, ALT_PIPELINE_PATH)
            joblib.dump(stage1_pipe, STAGE1_PIPELINE_PATH)

            self.is_loaded = True
            print(f"Model saved to {MODEL_PATH}")
            print(f"Vectorizer saved to {VECTORIZER_PATH}")
            print(f"Pipeline saved to {ALT_PIPELINE_PATH}")
            print(f"Stage1 pipeline saved to {STAGE1_PIPELINE_PATH}")
            return {"success": True, "details": f"Trained on {len(train_texts)} samples."}

        except Exception as e:
            print(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def _iter_samples_from_file(self, f, fmt: str) -> Iterable[Tuple[str, str]]:
        if fmt == "csv":
            rdr = csv.DictReader(f)
            for row in rdr:
                t = (row or {}).get("text") or ""
                l = (row or {}).get("label") or ""
                t = t.strip()
                l = l.strip()
                if t and l:
                    yield (t, l)
        elif fmt == "ndjson":
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = (obj or {}).get("text") or ""
                l = (obj or {}).get("label") or ""
                t = t.strip()
                l = l.strip()
                if t and l:
                    yield (t, l)
        elif fmt == "json_array":
            try:
                arr = json.load(f)
            except Exception:
                arr = []
            if isinstance(arr, list):
                for it in arr:
                    t = (it or {}).get("text") or ""
                    l = (it or {}).get("label") or ""
                    t = t.strip()
                    l = l.strip()
                    if t and l:
                        yield (t, l)
        else:
            return

    def train_model_incremental_from_path(
        self,
        path: str,
        fmt: str,
        batch_size: int = 1000,
        job_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_pipeline_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Incremental training for large datasets.
        fmt: csv | ndjson | json_array
        """
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"Dataset not found: {path}"}

            # First pass: collect labels + count
            labels_set = set()
            total = 0
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                for _, label in self._iter_samples_from_file(f, fmt):
                    labels_set.add(label)
                    total += 1
                    if job_hook and total % max(1000, batch_size) == 0:
                        job_hook({"phase": "scan", "processed": total})

            if not labels_set:
                return {"success": False, "error": "No training data provided"}

            classes = np.array(sorted(labels_set))

            # Incremental pipeline (fixed memory)
            self.vectorizer = HashingVectorizer(
                n_features=2**20,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                stop_words="english",
            )
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                random_state=42,
            )

            processed = 0
            batch_texts: List[str] = []
            batch_labels: List[str] = []

            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                for text, label in self._iter_samples_from_file(f, fmt):
                    batch_texts.append(text)
                    batch_labels.append(label)
                    if len(batch_texts) >= batch_size:
                        X = self.vectorizer.transform(batch_texts)
                        self.model.partial_fit(X, batch_labels, classes=classes)
                        processed += len(batch_texts)
                        batch_texts = []
                        batch_labels = []
                        if job_hook:
                            job_hook({"phase": "train", "processed": processed, "total": total})

            if batch_texts:
                X = self.vectorizer.transform(batch_texts)
                self.model.partial_fit(X, batch_labels, classes=classes)
                processed += len(batch_texts)
                if job_hook:
                    job_hook({"phase": "train", "processed": processed, "total": total})

            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)

            if save_pipeline_path:
                os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
                pipe = Pipeline([("tfidf", self.vectorizer), ("clf", self.model)])
                joblib.dump(pipe, save_pipeline_path)

            self.is_loaded = True
            return {
                "success": True,
                "details": f"Incremental trained on {processed} samples.",
                "total": total,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_document_type_by_rules(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        if not t.strip():
            return None
        print(f"[DEBUG] detect_document_type_by_rules: text_len={len(t)}")
        ui_tabs = ["overview","risks","entities","timeline","chat","comparison","report"]
        ui_cues = ["ask about your document","answer:","ask"]
        ui_hits = sum(1 for k in ui_tabs if k in t)
        ui_cue_hits = sum(1 for k in ui_cues if k in t)
        if ui_hits >= 3 and ui_cue_hits >= 1:
            print("[DEBUG] detect_document_type_by_rules: Application Interface Screenshot heuristic matched")
            return "Application Interface Screenshot"
        # Prefer explicit lease/rental cues over stamp paper headers.
        if "lease deed" in t:
            print("[DEBUG] detect_document_type_by_rules: Lease Deed heuristic matched")
            return "Lease Deed"
        property_cues = [
            "patta","பட்டா","survey","சர்வே","survey no","survey number",
            "tahsildar","தாசில்தார்","taluk","தாலுகா","village","கிராமம்",
            "tslr","town survey","fmb","field measurement book","ward","block",
            "layout approval","cmda","dtcp","schedule property","sub registrar","registration",
            "பதிவு","சப் ரெஜிஸ்ட்ரார்","உப பதிவு அலுவலகம்"
        ]
        if any(k in t for k in property_cues):
            print("[DEBUG] detect_document_type_by_rules: Property cues found -> Land Property")
            return "Land Property"
        tamil_present = re.search(r"[\u0B80-\u0BFF]", t) is not None
        stamp_present = ("non judicial" in t) or ("non-judicial" in t) or ("stamp paper" in t) or ("india non judicial" in t) or ("bond paper" in t) or ("தமிழ்நாடு" in t) or ("tamilnadu" in t) or ("tamil nadu" in t)
        if stamp_present and tamil_present:
            print("[DEBUG] detect_document_type_by_rules: Tamil stamp paper layout -> Land Property")
            return "Land Property"
        if "non judicial" in t or "non-judicial" in t or "stamp paper" in t or "india non judicial" in t or "bond paper" in t:
            print("[DEBUG] detect_document_type_by_rules: Non Judicial Stamp Paper heuristic matched")
            return "Non Judicial Stamp Paper"
        if "town survey land register" in t or "tslr" in t:
            print("[DEBUG] detect_document_type_by_rules: TSLR heuristic matched")
            return "TSLR"
        if "encumbrance certificate" in t or "certificate of encumbrance" in t:
            print("[DEBUG] detect_document_type_by_rules: Encumbrance Certificate heuristic matched")
            return "Encumbrance Certificate"
        if "employment contract" in t:
            print("[DEBUG] detect_document_type_by_rules: Employment Contract heuristic matched")
            return "Employment Contract"
        if "partnership deed" in t or "deed of partnership" in t:
            print("[DEBUG] detect_document_type_by_rules: Partnership Deed heuristic matched")
            return "Partnership Deed"
        if "services agreement" in t or "service agreement" in t:
            print("[DEBUG] detect_document_type_by_rules: Corporate Agreement (services) matched")
            return "Corporate Agreement"
        resume_cues = ["resume", "curriculum vitae", "professional summary", "work experience", "experience", "education", "skills", "projects"]
        cues_count = sum(1 for k in resume_cues if k in t)
        email_hit = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", t, flags=re.IGNORECASE) is not None
        phone_hit = re.search(r"\b(\+?\d{1,3}[-\s]?)?\d{10}\b", t) is not None
        if cues_count >= 2 or (email_hit and phone_hit and cues_count >= 1):
            print("[DEBUG] detect_document_type_by_rules: Resume heuristic matched")
            return "Resume"
        aadhaar_markers = ["aadhaar", "uidai", "unique identification", "आधार", "aadhar"]
        aadhaar_hits = sum(1 for k in aadhaar_markers if k in t)
        has_aadhaar_number = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text or "") is not None
        if aadhaar_hits >= 1 or has_aadhaar_number:
            if "pvc" in t:
                print("[DEBUG] detect_document_type_by_rules: Aadhaar PVC Card heuristic matched")
                return "Aadhaar PVC Card"
            print("[DEBUG] detect_document_type_by_rules: Aadhaar Card heuristic matched")
            return "Aadhaar Card"
        # OCR-tolerant PAN/e-PAN detection for noisy ID card scans.
        has_pan_number = re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", (text or "").upper()) is not None
        pan_like_markers = [
            "income tax department",
            "permanent account number",
            "govt. of india",
            "govt of india",
            "date of birth",
            "mother's name",
            "father's name",
            "income tax",
            "permanent account",
            "pan card",
            "pan no",
        ]
        pan_hits = sum(1 for k in pan_like_markers if k in t)
        if "e-pan" in t or "epan" in t:
            print("[DEBUG] detect_document_type_by_rules: e-PAN heuristic matched")
            return "e-PAN"
        if has_pan_number or pan_hits >= 2 or ("income tax" in t and pan_hits >= 1):
            print("[DEBUG] detect_document_type_by_rules: PAN Card heuristic matched")
            return "PAN Card"
        if "tax invoice" in t or "invoice" in t or "bill to" in t or "invoice no" in t:
            print("[DEBUG] detect_document_type_by_rules: Invoice heuristic matched")
            return "Invoice"
        if "licence" in t or "license" in t or "dl no" in t or "rto" in t:
            print("[DEBUG] detect_document_type_by_rules: Driving Licence heuristic matched")
            return "Driving Licence"
        if ("id card" in t or "identity card" in t or "employee id" in t or "id badge" in t) and not ("non judicial" in t or "non-judicial" in t or "stamp paper" in t):
            print("[DEBUG] detect_document_type_by_rules: ID Card heuristic matched")
            return "ID Card"
        if "bank statement" in t or "account statement" in t or "card account statement" in t:
            print("[DEBUG] detect_document_type_by_rules: Bank Statement heuristic matched")
            return "Bank Statement"
        if "passport" in t or "republic of india" in t:
            print("[DEBUG] detect_document_type_by_rules: Passport heuristic matched")
            return "Passport"
        if "income certificate" in t:
            print("[DEBUG] detect_document_type_by_rules: Income Certificate heuristic matched")
            return "Income Certificate"
        if "birth certificate" in t:
            print("[DEBUG] detect_document_type_by_rules: Birth Certificate heuristic matched")
            return "Birth Certificate"
        if "death certificate" in t:
            print("[DEBUG] detect_document_type_by_rules: Death Certificate heuristic matched")
            return "Death Certificate"
        if "marriage certificate" in t or "certificate of marriage" in t:
            print("[DEBUG] detect_document_type_by_rules: Marriage Certificate heuristic matched")
            return "Marriage Certificate"
        if "ration card" in t or "family card" in t:
            print("[DEBUG] detect_document_type_by_rules: Ration Card heuristic matched")
            return "Ration Card"
        if "marksheet" in t or "mark sheet" in t or "grade sheet" in t or "score card" in t:
            print("[DEBUG] detect_document_type_by_rules: Marksheet heuristic matched")
            return "Marksheet"
        if "admit card" in t or "hall ticket" in t:
            print("[DEBUG] detect_document_type_by_rules: Admit Card heuristic matched")
            return "Admit Card"
        if "transfer certificate" in t or "school leaving certificate" in t:
            print("[DEBUG] detect_document_type_by_rules: Transfer Certificate heuristic matched")
            return "Transfer Certificate"
        if "bonafide certificate" in t or "bonafide" in t:
            print("[DEBUG] detect_document_type_by_rules: Bonafide Certificate heuristic matched")
            return "Bonafide Certificate"
        if "experience certificate" in t:
            print("[DEBUG] detect_document_type_by_rules: Experience Certificate heuristic matched")
            return "Experience Certificate"
        if "salary certificate" in t or "salary slip" in t or "pay slip" in t:
            print("[DEBUG] detect_document_type_by_rules: Salary Certificate heuristic matched")
            return "Salary Certificate"
        if "gst registration certificate" in t or "gst certificate" in t or "gstin" in t:
            print("[DEBUG] detect_document_type_by_rules: GST Certificate heuristic matched")
            return "GST Certificate"
        if "court judgment" in t or "judgement" in t or "judgment" in t:
            print("[DEBUG] detect_document_type_by_rules: Court Judgment heuristic matched")
            return "Court Judgment"
        if "legal notice" in t or "notice under" in t:
            print("[DEBUG] detect_document_type_by_rules: Legal Notice heuristic matched")
            return "Legal Notice"
        if "residential rental agreement" in t:
            print("[DEBUG] detect_document_type_by_rules: Rental Agreement (residential) matched")
            return "Rental Agreement"
        if "rental agreement" in t:
            print("[DEBUG] detect_document_type_by_rules: Rental Agreement heuristic matched")
            return "Rental Agreement"
        if ("tenant" in t and "landlord" in t) and ("rent" in t or "lease" in t):
            print("[DEBUG] detect_document_type_by_rules: Rental Agreement keyword match")
            return "Rental Agreement"
        if ("lessor" in t and "lessee" in t) and ("rent" in t or "lease" in t):
            print("[DEBUG] detect_document_type_by_rules: Rental Agreement (lessor/lessee) match")
            return "Rental Agreement"
        if "gift deed" in t:
            print("[DEBUG] detect_document_type_by_rules: Gift Deed heuristic matched")
            return "Gift Deed"
        if ("donor" in t and "donee" in t) and ("gift" in t or "deed" in t):
            print("[DEBUG] detect_document_type_by_rules: Gift Deed (donor/donee) match")
            return "Gift Deed"
        resume_score = _resume_like_score(t)
        legal_markers = ["agreement", "deed", "parties", "witnesseth", "hereby", "clause", "schedule"]
        legal_hits = sum(1 for k in legal_markers if k in t)
        if resume_score >= 3 and legal_hits < 2:
            print("[DEBUG] detect_document_type_by_rules: Resume heuristic matched")
            return "Resume"
        for rule in _DOC_TYPE_RULES_COMPILED:
            all_ok = all(k in t for k in rule.get("all", []))
            any_list = rule.get("any", [])
            any_ok = True if not any_list else any(k in t for k in any_list)
            regex_list = rule.get("regex_compiled", [])
            regex_ok = True if not regex_list else any(rx.search(text or "") for rx in regex_list)
            if all_ok and any_ok and regex_ok:
                print(f"[DEBUG] detect_document_type_by_rules: Matched '{rule['label']}'")
                return rule["label"]
        print("[DEBUG] detect_document_type_by_rules: No rule match found")
        return None

    def predict_document_type(self, text: str) -> Tuple[str, float, str]:
        if text is None or not text.strip():
            print(f"[DEBUG] ML Service: Empty text.")
            return "Generic Legal Document", 0.0, MODEL_VERSION
            
        # 1. Long Document Strategy: Sliding Window
        # If text is long (> 3000 chars), use sliding window to aggregate votes
        if len(text) > 3000:
            res = self.predict_with_sliding_window(text)
            return res[0], res[1], MODEL_VERSION

        if not self.is_loaded and not self.bert_model:
            print(f"[DEBUG] ML Service: Models not loaded. Proceeding with rules and heuristics.")

        try:
            clean_text = re.sub(r'\s+', ' ', text).strip()[:5000]
            
            # Cache check
            h = hashlib.sha256(clean_text.encode("utf-8", errors="ignore")).hexdigest()
            cached = self._pred_cache.get(h)
            if cached:
                return cached

            # 2. Heuristic first: robust Resume detection purely from content
            low = clean_text.lower()
            resume_score = _resume_like_score(low)
            email_hit = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", clean_text, flags=re.IGNORECASE) is not None
            phone_hit = re.search(r"\b(\+?\d{1,3}[-\s]?)?\d{10}\b", clean_text) is not None
            strong_resume = (resume_score >= 2) or (resume_score >= 1 and (email_hit or phone_hit))
            if strong_resume:
                conf_pct = 50 + min(40, resume_score * 12) + (8 if (email_hit and phone_hit) else 0)
                conf_pct = max(55, min(95, conf_pct))
                out = ("Resume", conf_pct / 100.0, MODEL_VERSION)
                self._pred_cache[h] = out
                return out

            # 3. Rule-based match (Fast and precise for known patterns)
            rule_raw = self.detect_document_type_by_rules(clean_text)
            rule_pred = self.canonicalize_category(rule_raw)
            if rule_raw and rule_raw not in GENERIC_RULE_LABELS:
                out = (rule_raw, 0.92, MODEL_VERSION)
                self._pred_cache[h] = out
                return out

            # 4. Primary Classifier: BERT (If available)
            bert_res = self._predict_bert(clean_text)
            if bert_res and bert_res[1] > 0.85: # High confidence BERT match
                out = (self.canonicalize_category(bert_res[0]), bert_res[1], MODEL_VERSION)
                self._pred_cache[h] = out
                return out

            # 5. Baseline Fallback: TF-IDF + Logistic Regression
            if self.pipeline is not None:
                pred_raw = self.pipeline.predict([clean_text])[0]
                prediction = self.canonicalize_category(pred_raw)
                if hasattr(self.pipeline, "predict_proba"):
                    proba = self.pipeline.predict_proba([clean_text])[0]
                    conf_pct = int(np.max(proba) * 100)
                else:
                    conf_pct = 60
            elif self.model is not None:
                X_test = self.vectorizer.transform([clean_text])
                pred_raw = self.model.predict(X_test)[0]
                prediction = self.canonicalize_category(pred_raw)
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X_test)[0]
                    conf_pct = int(np.max(proba) * 100)
                else:
                    conf_pct = 60
            else:
                # Rule fallback if no ML models
                prediction = rule_pred or "Other Indian Legal Document"
                conf_pct = 50

            conf_pct = max(55, conf_pct)
            
            # Final confidence merging with BERT if BERT was low-confidence
            if bert_res and bert_res[1] > (conf_pct / 100.0):
                prediction = self.canonicalize_category(bert_res[0])
                conf_pct = int(bert_res[1] * 100)
 
            out = (prediction, min(100, conf_pct) / 100.0, MODEL_VERSION)
            self._pred_cache[h] = out
            return out
        except Exception as e:
            print(f"[DEBUG] ML Service: Prediction error: {e}")
            return "Generic Legal Document", 0.0, MODEL_VERSION

    def _predict_non_legal_subtype(self, clean_text: str) -> Tuple[str, float]:
        t = (clean_text or "").lower()
        if ("non judicial" in t) or ("non-judicial" in t) or ("stamp paper" in t) or ("india non judicial" in t) or ("bond paper" in t):
            return "Non Judicial Stamp Paper", 0.9
        best_label = "Other Non Legal Document"
        best_score = 0
        for label, hints in NON_LEGAL_HINTS.items():
            score = sum(1 for h in hints if h in t)
            if score > best_score:
                best_score = score
                best_label = label
        conf = 0.65 + min(0.30, best_score * 0.05)
        return best_label, min(conf, 0.95)

    def _semantic_fallback_label(
        self,
        text: str,
        current_label: str,
        current_conf_pct: int,
        candidate_labels: List[str],
    ) -> Tuple[str, int]:
        try:
            if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("DISABLE_HF") == "1":
                return current_label, int(current_conf_pct or 0)
        except Exception:
            return current_label, int(current_conf_pct or 0)
        try:
            from sentence_transformers import SentenceTransformer
            if getattr(self, "_semantic_model", None) is None:
                self._semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            labels = [l for l in candidate_labels if l in CATEGORY_DESCRIPTIONS]
            if not labels:
                return current_label, current_conf_pct
            if getattr(self, "_semantic_label_embeddings", None) is None or len(self._semantic_label_embeddings) != len(labels):
                self._semantic_label_embeddings = self._semantic_model.encode(
                    [CATEGORY_DESCRIPTIONS[l] for l in labels],
                    convert_to_numpy=True,
                )
            text_emb = self._semantic_model.encode([text], convert_to_numpy=True)[0]
            sims = []
            for idx in range(len(labels)):
                denom = (np.linalg.norm(text_emb) * np.linalg.norm(self._semantic_label_embeddings[idx])) + 1e-9
                sims.append(float(np.dot(text_emb, self._semantic_label_embeddings[idx]) / denom))
            best_idx = int(np.argmax(np.array(sims)))
            best_label = labels[best_idx]
            best_conf = int(max(55, min(98, round(sims[best_idx] * 100))))
            if best_conf > int(current_conf_pct or 0):
                return best_label, best_conf
        except Exception:
            pass
        return current_label, int(current_conf_pct or 0)

    def detect_anomaly(self, text: str, category: str, confidence_pct: int) -> Tuple[bool, str]:
        t = (text or "").strip().lower()
        reasons: List[str] = []

        if len(t) < 50:
            reasons.append("short_text")
        if int(confidence_pct or 0) < 60:
            reasons.append("low_confidence")

        kw = ["agreement", "deed", "parties", "witness", "signed", "dated", "hereby", "schedule", "property"]
        legal_like = {"Agreement", "Property document", "Sale Deed", "Lease Deed", "Rental Agreement"}
        if category in legal_like:
            count = sum(1 for k in kw if k in t)
            if count < 2:
                reasons.append("missing_keywords")

        req = REQUIRED_KEYWORDS.get(category) or []
        if req:
            req_count = sum(1 for k in req if k in t)
            if req_count < 1:
                reasons.append("missing_required_keywords")

        # Optional outlier signal: very sparse alpha tokens often indicates OCR garbage/noise.
        alpha_tokens = re.findall(r"[a-zA-Z]{2,}", t)
        if len(alpha_tokens) < 10 and len(t) > 120:
            reasons.append("noisy_or_sparse_tokens")

        if reasons:
            return True, ",".join(sorted(set(reasons)))
        return False, ""

    def detect_fraud(self, text: str) -> Tuple[bool, str]:
        t = (text or "")
        reasons: List[str] = []
        if re.search(r"\d{14,}", t):
            reasons.append("long_numeric_sequence")
        if re.search(r"lorem ipsum|dummy text|sample text|asdf", t, flags=re.IGNORECASE):
            reasons.append("dummy_pattern")
        if re.search(r"(X{4,}|\*{4,})", t):
            reasons.append("masking_pattern")
        if len(t) > 120:
            symbols = len(re.findall(r"[^A-Za-z0-9\s\.,:\-\/\(\)]", t))
            density = symbols / max(1, len(t))
            if density > 0.18:
                reasons.append("high_symbol_density")
        if len(t.strip()) > 300:
            tl = t.lower()
            sig_terms = ["signature", "signed by", "executant", "witness", "authorized signatory"]
            if not any(s in tl for s in sig_terms):
                reasons.append("No signature or witness reference found")
            seal_terms = ["seal", "stamp", "notarized", "attested", "sub-registrar", "tahsildar"]
            if not any(s in tl for s in seal_terms):
                reasons.append("No seal or stamp reference found")
        if re.search(r"\s{15,}", t) or re.search(r"(.)\1{10,}", t):
            reasons.append("Suspicious formatting anomaly detected")
        if 0 < len(t.strip()) < 150:
            reasons.append("Document appears incomplete or truncated")
        if reasons:
            return True, "; ".join(sorted(set(reasons)))
        return False, ""

    def save_for_auto_training(
        self,
        text: str,
        category: str,
        confidence_pct: int,
        anomaly_detected: bool,
        fraud_detected: bool = False,
    ) -> bool:
        try:
            stats = {
                "count": 0,
                "saved": 0,
                "last_added": None,
                "anomaly_skipped_count": 0,
                "fraud_skipped_count": 0,
                "low_conf_skipped_count": 0,
                "duplicate_skipped_count": 0,
            }
            if os.path.exists(AUTO_TRAIN_STATS_PATH):
                try:
                    with open(AUTO_TRAIN_STATS_PATH, "r", encoding="utf-8") as sf:
                        loaded = json.load(sf) or {}
                        if isinstance(loaded, dict):
                            stats.update(loaded)
                except Exception:
                    pass
            stats["count"] = int(stats.get("count", 0)) + 1

            if int(confidence_pct or 0) <= 85:
                stats["low_conf_skipped_count"] = int(stats.get("low_conf_skipped_count", 0)) + 1
                with open(AUTO_TRAIN_STATS_PATH, "w", encoding="utf-8") as sf:
                    json.dump(stats, sf, ensure_ascii=False, indent=2)
                return False
            if anomaly_detected:
                stats["anomaly_skipped_count"] = int(stats.get("anomaly_skipped_count", 0)) + 1
                with open(AUTO_TRAIN_STATS_PATH, "w", encoding="utf-8") as sf:
                    json.dump(stats, sf, ensure_ascii=False, indent=2)
                return False
            if fraud_detected:
                stats["fraud_skipped_count"] = int(stats.get("fraud_skipped_count", 0)) + 1
                with open(AUTO_TRAIN_STATS_PATH, "w", encoding="utf-8") as sf:
                    json.dump(stats, sf, ensure_ascii=False, indent=2)
                return False

            normalized_text = (text or "").strip()
            if len(normalized_text) < 80:
                return False

            os.makedirs(os.path.dirname(AUTO_TRAIN_PATH), exist_ok=True)
            payload = {
                "text": normalized_text[:5000],
                "category": category or "Unknown",
                "source": "auto",
                "confidence": int(confidence_pct or 0),
            }
            payload["sample_id"] = hashlib.sha1(
                f"{payload['category']}::{payload['text']}".encode("utf-8", errors="ignore")
            ).hexdigest()

            data = []
            if os.path.exists(AUTO_TRAIN_PATH):
                try:
                    with open(AUTO_TRAIN_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f) or []
                        if not isinstance(data, list):
                            data = []
                except Exception:
                    data = []

            existing_ids = {
                (d or {}).get("sample_id")
                for d in data
                if isinstance(d, dict) and (d or {}).get("sample_id")
            }
            if payload["sample_id"] in existing_ids:
                stats["duplicate_skipped_count"] = int(stats.get("duplicate_skipped_count", 0)) + 1
                with open(AUTO_TRAIN_STATS_PATH, "w", encoding="utf-8") as sf:
                    json.dump(stats, sf, ensure_ascii=False, indent=2)
                return False

            data.append(payload)
            with open(AUTO_TRAIN_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            stats["saved"] = int(stats.get("saved", 0)) + 1
            stats["last_added"] = int(time.time())
            with open(AUTO_TRAIN_STATS_PATH, "w", encoding="utf-8") as sf:
                json.dump(stats, sf, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def verify_document_ml(self, text: str) -> Dict[str, Any]:
        """
        Verify document legitimacy using ML confidence, heuristic rules, and layout consistency.
        """
        from backend.app.services import analysis_service
        safe_text = text or ""
        safe_text_lower = safe_text.lower()

        if not safe_text.strip():
            return {
                "marker": "Rejected",
                "ai_confidence": 0,
                "risk_level": "Unknown",
                "issues": ["Empty text", "Model could not classify document (Confidence: 0%)."],
                "predicted_category": "Unknown"
            }

        res = self.predict_document_type(safe_text)
        cat, conf = res[0], res[1]
        cat = self.canonicalize_category(cat)
        rule_cat = self.canonicalize_category(self.detect_document_type_by_rules(safe_text))
        
        # New: Numerical Layout Score
        layout_score = analysis_service.compute_layout_score(safe_text)

        print(f"[DEBUG] verify_document_ml: Initial Cat='{cat}', Conf={conf:.4f}, RuleCat='{rule_cat}', LayoutScore={layout_score:.2f}")

        # 1. DOCUMENT TYPE MATCHING WITH LEGAL INDICATOR GUARD
        legal_indicators = [
            "agreement","deed","affidavit","court","notary","jurisdiction",
            "petitioner","respondent","stamp duty","witness"
        ]
        indicator_count = sum(1 for k in legal_indicators if k in safe_text_lower)
        
        # Combined Confidence (ML + Layout)
        # Authentic legal docs have high structural consistency
        weighted_conf = (conf * 0.7) + (layout_score * 0.3)
        
        if rule_cat == "Resume" or cat == "Resume":
            cat = "Resume"
        elif rule_cat:
            if weighted_conf >= 0.70 and indicator_count >= 1:
                print(f"[DEBUG] verify_document_ml: Applying guarded rule override. '{cat}' -> '{rule_cat}'")
                cat = rule_cat
        elif cat in ["Generic Legal Document", "Unknown", "Other Non Legal Document"] and weighted_conf < 0.70:
            weighted_conf = max(weighted_conf, 0.40)

        ai_confidence = int(weighted_conf * 100)

        # 2. MARKER & RISK CALCULATION
        marker = "Verified"
        risk_level = "Low"
        issues = []

        if ai_confidence < 45:
            marker = "Rejected"
            risk_level = "Critical"
            issues.append(f"Model could not classify document (Confidence: {ai_confidence}%).")
        elif ai_confidence < 75:
            marker = "Flagged"
            risk_level = "High"
            issues.append(f"Low classification confidence ({ai_confidence}%) for category '{cat}'")
        elif ai_confidence < 88:
            risk_level = "Medium"

        print(f"[DEBUG] verify_document_ml: AI Confidence={ai_confidence}%, Marker='{marker}', Risk='{risk_level}'")

        # 3. HEURISTICS
        # Text Length
        text_len = len(safe_text.strip())
        if text_len < 150:
            if rule_cat:
                if risk_level == "Low":
                    risk_level = "Medium"
                issues.append(f"Document text is relatively short ({text_len} chars), but matched via rules.")
            else:
                marker = "Rejected"
                risk_level = "Critical"
                issues.append(f"Document text is too short ({text_len} chars) for reliable analysis.")
                ai_confidence = 0

        # Structure Check
        legal_keywords = ["agreement", "deed", "parties", "witness", "signed", "dated", "hereby", "schedule", "property"]
        keyword_count = sum(1 for k in legal_keywords if k in safe_text_lower)
        
        if "agreement" in cat.lower() or "deed" in cat.lower() or cat == "Sale Deed":
            if keyword_count < 3:
                risk_level = "High" if risk_level != "Critical" else "Critical"
                issues.append("Missing standard legal structure keywords.")
                marker = "Flagged"
                ai_confidence = max(25, ai_confidence - 20)
            elif keyword_count >= 6:
                # Strong legal structure found
                ai_confidence = min(100, ai_confidence + 10)

        # Non-legal safety: low confidence with no legal indicators -> Non-Legal Document
        if ai_confidence < 50 and indicator_count == 0 and cat not in {"Resume","Invoice","ID Card"}:
            cat = "Non-Legal Document"
            marker = "Verified"
            risk_level = "None"

        return {
            "marker": marker,
            "ai_confidence": ai_confidence,
            "risk_level": risk_level,
            "issues": issues,
            "predicted_category": cat,
            "model_version": MODEL_VERSION
        }

    def evaluate_dataset(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        try:
            if not texts or not labels:
                return {"success": False, "error": "No evaluation data provided"}
            if not self.is_loaded and not self.bert_model:
                return {"success": False, "error": "No model loaded for evaluation"}
                
            preds = []
            # Batch processing for evaluation
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                if self.bert_model:
                    batch_res = self._predict_bert_batch(batch)
                    preds.extend([self.canonicalize_category(r[0]) for r in batch_res])
                else:
                    X = self.vectorizer.transform(batch)
                    preds.extend(self.model.predict(X))
            
            acc = float(accuracy_score(labels, preds))
            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(labels, preds).tolist()
            
            return {
                "success": True, 
                "accuracy": acc, 
                "report": report, 
                "confusion_matrix": cm,
                "model_version": MODEL_VERSION,
                "engine": "BERT" if self.bert_model else "TF-IDF"
            }
        except Exception as e:
            print(f"[DocumentClassifier] Evaluation failed: {e}")
            return {"success": False, "error": str(e)}

classifier = DocumentClassifier()
