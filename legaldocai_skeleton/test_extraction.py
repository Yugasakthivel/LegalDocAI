from app.services.llm_service import analyze_legal_document
from app.models.schemas import AnalysisResult, ClauseInfo, PartyInfo, EntityData

sample_text = """
RENTAL AGREEMENT

This Rental Agreement is made on 15-01-2024 between:

LANDLORD: Mr. Rajesh Kumar, S/o Late Ramesh Kumar, residing at 45 MG Road, Bangalore - 560001

TENANT: Ms. Priya Sharma, D/o Mr. Vijay Sharma, residing at 12 Brigade Road, Bangalore - 560025

PROPERTY: Flat No. 301, Survey No. 45/2A, Green Valley Apartments, Whitefield, Bangalore - 560066

TERMS:
1. RENT: The monthly rent is Rs. 25,000/- (Rupees Twenty Five Thousand only)
2. SECURITY DEPOSIT: Rs. 1,00,000/- (Rupees One Lakh only)
3. DURATION: 11 months starting from 01-02-2024 to 31-12-2024
4. PAYMENT: Rent to be paid on or before 5th of every month
5. TERMINATION: Either party may terminate with 2 months notice
6. MAINTENANCE: Tenant responsible for minor repairs, landlord for major repairs
7. JURISDICTION: Courts in Bangalore shall have exclusive jurisdiction

Witnesses:
1. Mr. Anil Verma
2. Mrs. Sunita Rao
"""

print("Testing enhanced entity extraction and clause identification...")
print("=" * 80)

result = analyze_legal_document(sample_text)

print(f"\nDocument Type: {result.document_type}")
print(f"\nSummary: {result.summary_simple}")

print("\n\nENTITIES EXTRACTED:")
print("-" * 80)

print("\nParties:")
for party in result.entities.parties:
    print(f"  - {party.name} ({party.role if party.role else 'No role specified'})")

print(f"\nDates: {result.entities.dates}")
print(f"\nAmounts: {result.entities.amounts}")
print(f"\nProperties: {result.entities.properties}")
print(f"\nSurvey Numbers: {result.entities.survey_numbers}")
print(f"\nRegistration Numbers: {result.entities.registration_numbers}")

print("\n\nCLAUSES IDENTIFIED:")
print("-" * 80)
for clause in result.clauses:
    print(f"\n[{clause.category.upper()}]")
    print(f"  {clause.content}")

print("\n\nRISK ANALYSIS:")
print("-" * 80)
print(f"Level: {result.risk_analysis.level}")
for detail in result.risk_analysis.details:
    print(f"  - {detail}")

print(f"\n\nConfidence Score: {result.confidence_score}")
print("\n" + "=" * 80)
print("Test completed successfully!")
