import json
import re
from typing import Optional
from openai import OpenAI
from app.models.schemas import AnalysisResult, EntityData, RiskAnalysis
from app.config import settings

client = None
if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-your"):
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize OpenAI client: {e}")

INDIAN_LEGAL_DOCUMENT_TYPES = [
    "Sale Deed", "Lease Deed", "Rental Agreement", "Gift Deed", 
    "Patta", "Town Survey Land Register (TSLR)", "Encumbrance Certificate",
    "Court Judgment", "Court Order", "Legal Notice", "Affidavit",
    "Employment Contract", "Partnership Deed", "Corporate Agreement",
    "Power of Attorney", "Will", "Trust Deed", "Mortgage Deed",
    "Unknown Legal Document"
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
    "parties": ["list of people/organizations involved"],
    "dates": ["important dates in DD-MM-YYYY or text format"],
    "amounts": ["monetary amounts with Rs/₹ symbol"],
    "properties": ["property details, survey numbers, addresses"],
    "case_numbers": ["court case numbers, registration numbers"],
    "courts": ["courts or authorities mentioned"]
  }},
  "clauses": ["list of important clauses like ownership, payment, termination, jurisdiction"],
  "key_terms": ["important legal terms and concepts"],
  "risk_analysis": {{
    "level": "<LOW, MEDIUM, or HIGH>",
    "details": ["specific risks, missing clauses, vague language, issues found"]
  }},
  "compliance_flags": ["Indian legal compliance items: registration, stamp duty, witnesses, etc."],
  "confidence_score": <0.0 to 1.0>
}}

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
        
        return AnalysisResult(
            document_type=analysis_data.get("document_type", "Unknown Legal Document"),
            summary_simple=analysis_data.get("summary_simple", "Analysis completed"),
            entities=EntityData(**analysis_data.get("entities", {})),
            clauses=analysis_data.get("clauses", []),
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
            parties=["LLM integration required"],
            dates=[],
            amounts=[],
            properties=[],
            case_numbers=[],
            courts=[]
        ),
        clauses=["LLM integration required for clause detection"],
        key_terms=[],
        risk_analysis=RiskAnalysis(
            level="UNKNOWN",
            details=["Configure OpenAI API key to enable risk analysis"]
        ),
        compliance_flags=["LLM integration required for compliance checking"],
        confidence_score=0.0
    )
