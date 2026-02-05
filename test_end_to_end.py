import requests
from PIL import Image, ImageDraw, ImageFont
import io

BASE = "http://127.0.0.1:8000"

def make_test_image():
    img = Image.new("RGB", (900, 500), (255, 255, 255))
    d = ImageDraw.Draw(img)
    text = (
        "RENTAL AGREEMENT\n"
        "Term: Jan 1 2025 to Dec 31 2025\n"
        "Rent: Rs 10,000 per month\n"
        "Access: Landlord may enter with 24h notice\n"
        "Signature: Authorized Signatory John Doe\n"
        "Governing Law: Indian Contract Act\n"
    )
    d.text((20, 20), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def upload_document():
    img_buf = make_test_image()
    files = {"file": ("test_agreement.png", img_buf, "image/png")}
    res = requests.post(f"{BASE}/api/upload", files=files)
    res.raise_for_status()
    data = res.json()
    return data["doc_id"]

def run_pipeline(doc_id):
    data = {"doc_id": doc_id}
    res = requests.post(f"{BASE}/api/pipeline/full", data=data)
    res.raise_for_status()
    return res.json()

def ask_qna(doc_id, question):
    res = requests.post(f"{BASE}/api/search/rag", params={"doc_id": doc_id, "question": question})
    # rag may 404 if vector DB disabled; still try
    try:
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e), "status_code": res.status_code, "text": res.text}

if __name__ == "__main__":
    print("Uploading test document...")
    doc_id = upload_document()
    print("doc_id:", doc_id)
    print("Running full pipeline...")
    out = run_pipeline(doc_id)
    summ = out.get("summary", {})
    kf = summ.get("key_facts", {})
    ai_conf = summ.get("analytics", {}).get("ai_confidence")
    cri = summ.get("analytics", {}).get("combined_risk_index")
    print("Key Facts:", kf)
    print("AI Confidence:", ai_conf, "Combined Risk Index:", cri)
    print("Asking QnA...")
    ans = ask_qna(doc_id, "What is the monthly rent?")
    print("QnA:", ans)
