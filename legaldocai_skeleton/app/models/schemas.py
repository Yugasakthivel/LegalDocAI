from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PartyInfo(BaseModel):
    name: str = Field(description="Name of party")
    role: Optional[str] = Field(default=None, description="Role: buyer, seller, petitioner, respondent, employer, employee, etc.")

class EntityData(BaseModel):
    parties: List[PartyInfo] = Field(default_factory=list, description="Parties involved with their roles")
    dates: List[str] = Field(default_factory=list, description="Important dates")
    amounts: List[str] = Field(default_factory=list, description="Monetary amounts (₹)")
    properties: List[str] = Field(default_factory=list, description="Property details (survey numbers, addresses)")
    survey_numbers: List[str] = Field(default_factory=list, description="Survey numbers and plot identifiers")
    case_numbers: List[str] = Field(default_factory=list, description="Case numbers")
    registration_numbers: List[str] = Field(default_factory=list, description="Registration numbers")
    courts: List[str] = Field(default_factory=list, description="Courts or authorities mentioned")

class ClauseInfo(BaseModel):
    category: str = Field(description="Clause category: ownership, payment, termination, rights_obligations, penalties, jurisdiction, other")
    content: str = Field(description="Clause text or summary")

class RiskAnalysis(BaseModel):
    level: str = Field(description="Risk level: LOW, MEDIUM, HIGH, UNKNOWN")
    details: List[str] = Field(default_factory=list, description="Risk details and recommendations")

class AnalysisResult(BaseModel):
    document_type: str = Field(description="Type of legal document")
    summary_simple: str = Field(description="Simple English explanation of the document")
    entities: EntityData = Field(description="Extracted entities from the document")
    clauses: List[ClauseInfo] = Field(default_factory=list, description="Identified and categorized clauses")
    key_terms: List[str] = Field(default_factory=list, description="Important legal terms")
    risk_analysis: RiskAnalysis = Field(description="Risk assessment")
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance issues or requirements")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Analysis confidence (0-1)")

class UploadResponse(BaseModel):
    status: str = Field(description="Response status: success or error")
    filename: str = Field(description="Uploaded filename")
    file_type: str = Field(description="Detected file type")
    extracted_text: str = Field(description="Text extracted from the document")
    analysis: AnalysisResult = Field(description="AI analysis results")
    disclaimer: str = Field(
        default="This analysis is for educational purposes only and does not constitute legal advice.",
        description="Legal disclaimer"
    )

class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Error status")
    message: str = Field(description="Error message")
    detail: str = Field(default="", description="Detailed error information")
