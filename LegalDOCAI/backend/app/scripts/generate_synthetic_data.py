import os
import json
import random
from typing import List

# Define output path
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../legal_dataset")
OUTPUT_FILE = os.path.join(DATA_DIR, "training_data.json")

# Categories matching ml_service.py
CATEGORIES = [
    "Sale Deed", "Lease Deed", "Rental Agreement", "Gift Deed", "Patta",
    "Town Survey Land Register (TSLR)", "Encumbrance Certificate",
    "Court Judgment", "Court Order", "Legal Notice", "Affidavit",
    "Petition", "Employment Contract", "Partnership Deed",
    "MOA / AOA", "Company Agreement", "Regulatory Filing"
]

# Simple templates to generate somewhat realistic text for each category
TEMPLATES = {
    "Sale Deed": [
        "This DEED OF SALE is made at {city} on this {date} day of {month}, {year}, between {vendor} (Vendor) and {vendee} (Vendee). The Vendor hereby sells, conveys and transfers the property...",
        "ABSOLUTE SALE DEED. The Vendor is the absolute owner of the schedule property... In consideration of Rs. {amount} paid by the Purchaser...",
        "THIS INDENTURE OF SALE WITNESSETH that in consideration of the sum of...",
    ],
    "Lease Deed": [
        "This LEASE DEED is executed on {date} by {lessor} (Lessor) in favor of {lessee} (Lessee). The Lessor hereby grants lease of the premises for a period of {years} years...",
        "DEED OF LEASE. The Lessee shall pay a monthly rent of Rs. {rent}. The lease term is for 11 months...",
        "The Lessor agrees to let out the detailed property on lease to the Lessee...",
    ],
    "Rental Agreement": [
        "RENTAL AGREEMENT entered into on {date} between Landlord {landlord} and Tenant {tenant}. The Tenant agrees to pay rent of Rs. {rent} per month...",
        "This Agreement of Tenancy is made... The Tenant shall use the premises for residential purposes only...",
        "Tenancy Agreement. Witnesseth as follows: The Landlord hereby lets the apartment...",
    ],
    "Gift Deed": [
        "This DEED OF GIFT is made by {donor} (Donor) to {donee} (Donee). Out of natural love and affection, the Donor hereby gifts the property...",
        "I, {donor}, do hereby transfer by way of Gift, the entire schedule property to my son {donee}...",
        "GIFT DEED. The Donor is the absolute owner... and wishes to gift the same...",
    ],
    "Patta": [
        "GOVERNMENT OF TAMIL NADU - REVENUE DEPARTMENT. Patta No: {patta_no}. Owner: {owner}. Survey No: {survey_no}...",
        "Extract from the Permanent Land Register. Patta Passbook. District: {district}. Taluk: {taluk}...",
        "Ryotwari Patta. Name of Pattadar: {owner}. Wet/Dry Land...",
    ],
    "Town Survey Land Register (TSLR)": [
        "Town Survey Land Register (TSLR) Extract. Ward: {ward}. Block: {block}. Town Survey No: {ts_no}...",
        "Land Record Extract (TSLR). Classification regarding ownership: Private Holding...",
        "Municipality / Corporation Record. TSLR Number...",
    ],
    "Encumbrance Certificate": [
        "ENCUMBRANCE CERTIFICATE. Application No: {app_no}. Statement of Encumbrance on Property...",
        "Nil Encumbrance Certificate. From {date1} to {date2}. Description of Property...",
        "Registration Department. EC. Document No...",
    ],
    "Court Judgment": [
        "IN THE HIGH COURT OF JUDICATURE AT {city}. Present: The Hon'ble Mr. Justice {judge}. W.P. No. {case_no}...",
        "JUDGMENT. The petitioner has filed this writ petition under Article 226 of the Constitution...",
        "ORDER. This appeal arises out of the order passed by the learned Single Judge...",
    ],
    "Court Order": [
        "INTERIM ORDER. It is hereby ordered that the status quo be maintained...",
        "ORDER SHEET. Case No: {case_no}. Proceeding dated {date}...",
        "DISMISSAL ORDER. The court finds no merit in the petition and hereby dismisses it...",
    ],
    "Legal Notice": [
        "LEGAL NOTICE under Section 138 of Negotiable Instruments Act. My client, {client}, instructs me to state...",
        "NOTICE. By Speed Post AD. To: {recipient}. Sub: Demand for payment of outstanding dues...",
        "ADVOCATE NOTICE. Please take notice that you are hereby called upon to...",
    ],
    "Affidavit": [
        "AFFIDAVIT. I, {deponent}, son of {father}, aged {age}, residing at {address}, do hereby solemnly affirm and declare...",
        "SWORN AFFIDAVIT. Before the Notary Public. I, the deponent named above...",
        "I hereby verify that the contents of this affidavit are true to the best of my knowledge...",
    ],
    "Petition": [
        "WRIT PETITION under Article 226. In the matter of... Petitioner vs Respondent...",
        "Plaint in O.S. No. {case_no}. Suit for Permanent Injunction...",
        "Original Petition. Prayer: To grant letters of administration...",
    ],
    "Employment Contract": [
        "EMPLOYMENT AGREEMENT. This contract is made between {employer} (Company) and {employee} (Employee)...",
        "Offer of Appointment. We are pleased to offer you the position of...",
        "Contract of Service. Terms and conditions of employment including salary, probation, and termination...",
    ],
    "Partnership Deed": [
        "DEED OF PARTNERSHIP. This deed of partnership is made this {date} between {partner1} and {partner2}...",
        "We, the undersigned, hereby form a partnership under the name and style of...",
        "Partnership Agreement. Profit and loss sharing ratio...",
    ],
    "MOA / AOA": [
        "THE COMPANIES ACT, 2013. MEMORANDUM OF ASSOCIATION of {company_name} Private Limited...",
        "ARTICLES OF ASSOCIATION. Interpretation: 'The Company' means...",
        "MOA and AOA. Objects clause: To carry on the business of...",
    ],
    "Company Agreement": [
        "SHAREHOLDERS AGREEMENT. This agreement is entered into by and between the Shareholders...",
        "Share Purchase Agreement. The Sellers agree to sell and the Buyers agree to purchase the Sale Shares...",
        "Joint Venture Agreement. Between Company A and Company B...",
    ],
    "Regulatory Filing": [
        "FORM NO. MGT-7. Annual Return. Pursuant to Section 92(1) of the Companies Act, 2013...",
        "GST Registration Certificate. Form GST REG-06...",
        "Income Tax Return Verification Form. ITR-V...",
    ]
}

# Generic words to fill placeholders
CITIES = ["Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad"]
NAMES = ["John Doe", "Jane Smith", "Kumar", "Ravi", "Suresh", "Priya", "ABC Corp", "XYZ Ltd"]
MONTHS = ["January", "February", "March", "April", "May", "June"]

def fill_template(s: str) -> str:
    s = s.replace("{city}", random.choice(CITIES))
    s = s.replace("{date}", str(random.randint(1, 28)))
    s = s.replace("{month}", random.choice(MONTHS))
    s = s.replace("{year}", str(random.randint(2010, 2025)))
    s = s.replace("{vendor}", random.choice(NAMES))
    s = s.replace("{vendee}", random.choice(NAMES))
    s = s.replace("{amount}", str(random.randint(100000, 10000000)))
    s = s.replace("{lessor}", random.choice(NAMES))
    s = s.replace("{lessee}", random.choice(NAMES))
    s = s.replace("{years}", str(random.randint(1, 99)))
    s = s.replace("{rent}", str(random.randint(5000, 50000)))
    s = s.replace("{landlord}", random.choice(NAMES))
    s = s.replace("{tenant}", random.choice(NAMES))
    s = s.replace("{donor}", random.choice(NAMES))
    s = s.replace("{donee}", random.choice(NAMES))
    s = s.replace("{patta_no}", str(random.randint(100, 9999)))
    s = s.replace("{owner}", random.choice(NAMES))
    s = s.replace("{survey_no}", f"{random.randint(1,500)}/{random.randint(1,10)}")
    s = s.replace("{district}", random.choice(CITIES))
    s = s.replace("{taluk}", "Taluk " + random.choice(CITIES))
    s = s.replace("{ward}", "Ward " + str(random.randint(1, 20)))
    s = s.replace("{block}", "Block " + str(random.randint(1, 50)))
    s = s.replace("{ts_no}", str(random.randint(100, 9999)))
    s = s.replace("{app_no}", str(random.randint(10000, 99999)))
    s = s.replace("{date1}", "01-01-2010")
    s = s.replace("{date2}", "01-01-2025")
    s = s.replace("{judge}", random.choice(NAMES))
    s = s.replace("{case_no}", str(random.randint(1000, 9999)) + "/2023")
    s = s.replace("{client}", random.choice(NAMES))
    s = s.replace("{recipient}", random.choice(NAMES))
    s = s.replace("{deponent}", random.choice(NAMES))
    s = s.replace("{father}", random.choice(NAMES))
    s = s.replace("{age}", str(random.randint(25, 70)))
    s = s.replace("{address}", "No 12, Gandhi Road, " + random.choice(CITIES))
    s = s.replace("{employer}", random.choice(NAMES))
    s = s.replace("{employee}", random.choice(NAMES))
    s = s.replace("{partner1}", random.choice(NAMES))
    s = s.replace("{partner2}", random.choice(NAMES))
    s = s.replace("{company_name}", random.choice(NAMES))
    return s

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

    data_entries = []
    
    # Generate samples
    print("Generating synthetic data...")
    for cat in CATEGORIES:
        # Get templates for this category (or use generic if missing)
        temps = TEMPLATES.get(cat, ["This is a legal document related to " + cat + "."])
        
        # Create X variations per template
        for t in temps:
            for _ in range(20): # Generate 20 variations of each template
                text = fill_template(t)
                # Add some random noise or variety
                text += " " + " ".join([random.choice(["Terms", "Conditions", "Witness", "Signed", "Dated"]) for _ in range(5)])
                data_entries.append({"text": text, "category": cat})
    
    # Shuffle
    random.shuffle(data_entries)
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data_entries, f, indent=2)
    
    print(f"✅ Generated {len(data_entries)} training samples in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
