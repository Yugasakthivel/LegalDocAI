import json
import re
from typing import Optional
from openai import OpenAI
from app.models.schemas import AnalysisResult, EntityData, RiskAnalysis, PartyInfo, ClauseInfo
from app.config import settings

client = None
if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-your"):
    try:
        import httpx
        # Explicitly create httpx client to avoid argument mismatch issues (e.g. proxies) in some versions
        http_client = httpx.Client(trust_env=False)
        client = OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)
    except Exception as e:
        print(f"Warning: Failed to initialize OpenAI client: {e}")

INDIAN_LEGAL_DOCUMENT_TYPES = [
    "Sale Deed", "Lease Deed", "Rental Agreement", "Gift Deed",
    "Patta", "Town Survey Land Register (TSLR)", "Encumbrance Certificate",
    "Court Judgment", "Court Order", "Legal Notice", "Affidavit", "Petition",
    "Employment Contract", "Partnership Deed", "MOA / AOA", "Company Agreement", "Regulatory Filing",
    "Power of Attorney", "Will", "Trust Deed", "Mortgage Deed",
    "Generic Legal Document", "Unknown Legal Document"
]

def analyze_legal_document(text: str, document_type: str = "Unknown") -> AnalysisResult:
    """
    Analyze legal document using OpenAI LLM.
    
    Args:
        text: Extracted document text
        document_type: Hint for document type (optional)
    
    Returns:
        AnalysisResult with structured analysis
    """
    
    if not client:
        return _placeholder_analysis()
    
    if not text or len(text.strip()) < 50:
        return _placeholder_analysis("Document text is too short for analysis")
    
    text_preview = text[:4000]
    
    prompt = f"""You are an expert AI legal assistant specializing in Indian legal documents.

Analyze the following Indian legal document and provide a structured JSON response.

Document text:
{text_preview}

Provide analysis in this EXACT JSON format:
{{
  "document_type": "<one of: {', '.join(INDIAN_LEGAL_DOCUMENT_TYPES)}>",
  "summary_simple": "<explain in simple English what this document is and its purpose>",
  "entities": {{
    "parties": [
      {{"name": "Party Name", "role": "buyer/seller/petitioner/respondent/employer/employee/landlord/tenant/etc"}}
    ],
    "dates": ["important dates in DD-MM-YYYY or text format"],
    "amounts": ["monetary amounts with Rs/₹ symbol and context"],
    "properties": ["property details with full addresses"],
    "survey_numbers": ["survey numbers, plot numbers, sy.no formats"],
    "case_numbers": ["court case numbers"],
    "registration_numbers": ["document registration numbers"],
    "courts": ["courts or authorities mentioned"]
  }},
  "clauses": [
    {{
      "category": "<ownership/payment/termination/rights_obligations/penalties/jurisdiction/other>",
      "content": "<clause text or summary>"
    }}
  ],
  "key_terms": ["important legal terms and concepts"],
  "risk_analysis": {{
    "level": "<LOW, MEDIUM, or HIGH>",
    "details": ["specific risks, missing clauses, vague language, issues found"]
  }},
  "compliance_flags": ["Indian legal compliance items: registration, stamp duty, witnesses, etc."],
  "confidence_score": <0.0 to 1.0>
}}

IMPORTANT EXTRACTION GUIDELINES:
1. PARTIES: Extract all parties with their specific roles (buyer, seller, landlord, tenant, employer, employee, petitioner, respondent, etc.)
2. DATES: Extract execution date, effective date, expiry date, court dates
3. AMOUNTS: Include context (e.g., "₹50,00,000 - Sale consideration", "₹10,000/month - Rent")
4. SURVEY NUMBERS: Extract all survey numbers, plot numbers in formats like "Sy.No. 123/4", "Plot No. 45"
5. CLAUSES: Categorize each clause:
   - ownership: Transfer of title, ownership rights, possession clauses
   - payment: Payment terms, installments, penalties for delay
   - termination: Termination conditions, notice periods, exit clauses
   - rights_obligations: Rights and duties of parties
   - penalties: Late fees, breach penalties, liquidated damages
   - jurisdiction: Dispute resolution, applicable law, court jurisdiction
   - other: Any other important clauses

Return ONLY valid JSON, no additional text."""
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Indian legal document analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        analysis_data = json.loads(content)
        
        entities_data = analysis_data.get("entities", {})
        parties_raw = entities_data.get("parties", [])
        parties = []
        for party in parties_raw:
            if isinstance(party, dict):
                parties.append(PartyInfo(**party))
            elif isinstance(party, str):
                parties.append(PartyInfo(name=party, role=None))
        entities_data["parties"] = parties
        
        clauses_raw = analysis_data.get("clauses", [])
        clauses = []
        for clause in clauses_raw:
            if isinstance(clause, dict):
                clauses.append(ClauseInfo(**clause))
            elif isinstance(clause, str):
                clauses.append(ClauseInfo(category="other", content=clause))
        
        return AnalysisResult(
            document_type=analysis_data.get("document_type", "Unknown Legal Document"),
            summary_simple=analysis_data.get("summary_simple", "Analysis completed"),
            entities=EntityData(**entities_data),
            clauses=clauses,
            key_terms=analysis_data.get("key_terms", []),
            risk_analysis=RiskAnalysis(**analysis_data.get("risk_analysis", {"level": "UNKNOWN", "details": []})),
            compliance_flags=analysis_data.get("compliance_flags", []),
            confidence_score=float(analysis_data.get("confidence_score", 0.7))
        )
    
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return _placeholder_analysis(f"LLM response parsing failed: {e}")
    
    except Exception as e:
        print(f"LLM analysis error: {e}")
        return _placeholder_analysis(f"LLM analysis failed: {str(e)}")

def _placeholder_analysis(reason: str = "OpenAI API key not configured") -> AnalysisResult:
    """Fallback placeholder analysis when LLM is unavailable"""
    return AnalysisResult(
        document_type="Unknown (LLM not available)",
        summary_simple=f"LLM analysis unavailable. Reason: {reason}. Please configure OPENAI_API_KEY in .env file.",
        entities=EntityData(
            parties=[PartyInfo(name="LLM integration required", role=None)],
            dates=[],
            amounts=[],
            properties=[],
            survey_numbers=[],
            case_numbers=[],
            registration_numbers=[],
            courts=[]
        ),
        clauses=[ClauseInfo(category="other", content="LLM integration required for clause detection")],
        key_terms=[],
        risk_analysis=RiskAnalysis(
            level="UNKNOWN",
            details=["Configure OpenAI API key to enable risk analysis"]
        ),
        compliance_flags=["LLM integration required for compliance checking"],
        confidence_score=0.0
    )
