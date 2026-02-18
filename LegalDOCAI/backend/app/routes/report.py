from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from backend.app.database import documents_collection as collection
from backend.app.core.deps import get_openai_client
from backend.app.core.config import OPENAI_MODEL
import json
try:
    import fitz
except Exception:
    fitz = None

router = APIRouter(tags=["report"])

@router.get("/html")
async def report_html(doc_id: str, law: str = None):
    d = collection.find_one({"doc_id": doc_id}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="Document not found.")
    analytics = d.get("analytics", {})
    results = d.get("results", [])
    clause_risk = analytics.get("clause_risk", {})
    timeline = analytics.get("timeline", [])
    verification = {
        "marker": analytics.get("verified_marker", ""),
        "ai_confidence": analytics.get("ai_confidence", None)
    }
    compliance = None
    if law:
        try:
            prompt = (
                "Check the document against the specified regulation. "
                "Return strict JSON: {\"law\":\"...\",\"compliant\":true/false,"
                "\"violations\":[{\"rule\":\"...\",\"detail\":\"...\",\"highlight\":\"...\"}],"
                "\"recommendations\":[\"...\"],\"confidence\":0-100}. "
                "Law:\n" + law + "\n\nDocument:\n" + (d.get("combined_text","")[:6000])
            )
            resp = get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.1,
                max_tokens=600
            )
            raw = resp.choices[0].message.content or ""
            s = raw.find("{"); e = raw.rfind("}") + 1
            compliance = json.loads(raw[s:e]) if s != -1 and e > s else None
        except Exception:
            compliance = None
    def html_escape(t: str) -> str:
        return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    def list_items(items):
        return "".join(f"<li>{html_escape(str(i))}</li>" for i in items)
    clause_rows = ""
    for c in clause_risk.get("clauses", []):
        clause_rows += f"<tr><td>{html_escape(c.get('clause',''))}</td><td>{'Yes' if c.get('present') else 'No'}</td><td>{html_escape(c.get('risk',''))}</td><td>{html_escape(c.get('reason',''))}</td><td>{html_escape(c.get('highlight',''))}</td></tr>"
    timeline_items = "".join(f"<li><strong>{html_escape(ev.get('date',''))}</strong> — {html_escape(ev.get('event',''))}</li>" for ev in timeline)
    pages_preview = "".join(f"<details><summary>Page {r.get('page')}</summary><pre>{html_escape((r.get('text') or '')[:2000])}</pre></details>" for r in results[:10])
    compliance_block = ""
    if compliance:
        violations = compliance.get("violations", [])
        recommendations = compliance.get("recommendations", [])
        compliance_block = f"""
        <h2>Regulatory Compliance: {html_escape(compliance.get('law',''))}</h2>
        <p>Compliant: <strong>{'Yes' if compliance.get('compliant') else 'No'}</strong></p>
        <p>Confidence: {html_escape(str(compliance.get('confidence','')))}</p>
        <h3>Violations</h3>
        <ul>{list_items([f"{v.get('rule','')}: {v.get('detail','')}" for v in violations])}</ul>
        <h3>Recommendations</h3>
        <ul>{list_items(recommendations)}</ul>
        """
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>LegalDocAI Report — {html_escape(d.get('filename',''))}</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 16px; }}
        h1,h2,h3 {{ margin: 8px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background: #f5f5f5; }}
        .metric {{ display: inline-block; margin-right: 12px; }}
        pre {{ background: #f9f9f9; padding: 8px; white-space: pre-wrap; }}
        details {{ margin-bottom: 6px; }}
      </style>
    </head>
    <body>
      <h1>LegalDocAI Report</h1>
      <p><strong>File:</strong> {html_escape(d.get('filename',''))}</p>
      <div>
        <span class="metric"><strong>Verification:</strong> {html_escape(verification.get('marker',''))}</span>
        <span class="metric"><strong>AI Confidence:</strong> {html_escape(str(verification.get('ai_confidence')))}</span>
        <span class="metric"><strong>Legality Score:</strong> {html_escape(str(analytics.get('legality_score',0)))}</span>
        <span class="metric"><strong>Type:</strong> {html_escape(analytics.get('document_type','Unknown'))}</span>
      </div>
      <h2>Summary</h2>
      <p>{html_escape(analytics.get('summary',''))}</p>
      <h2>Clause Presence</h2>
      <table>
        <thead><tr><th>Clause</th><th>Present</th><th>Risk</th><th>Reason</th><th>Highlight</th></tr></thead>
        <tbody>{clause_rows}</tbody>
      </table>
      <h2>Timeline</h2>
      <ul>{timeline_items}</ul>
      {compliance_block}
      <h2>Document Content Preview</h2>
      {pages_preview}
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

@router.get("/pdf")
async def report_pdf(doc_id: str):
    if not fitz:
        raise HTTPException(status_code=503, detail="PDF generation unavailable (PyMuPDF not installed)")
    d = collection.find_one({"doc_id": doc_id}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    analytics = d.get("analytics", {})
    filename = d.get("filename", "document")
    
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text((50, 50), f"LegalDocAI Report: {filename}", fontsize=16, color=(0, 0, 0))
    
    # Metrics
    y = 80
    score = analytics.get("combined_risk_index", 0)
    page.insert_text((50, y), f"Risk Score: {score}/100", fontsize=12)
    y += 20
    doc_type = analytics.get("document_type", "Unknown")
    page.insert_text((50, y), f"Type: {doc_type}", fontsize=12)
    
    # Summary
    y += 40
    page.insert_text((50, y), "Summary:", fontsize=14, color=(0, 0, 1))
    y += 20
    summary = analytics.get("summary", "No summary available.")
    rect = fitz.Rect(50, y, 550, y + 200)
    rc = page.insert_textbox(rect, summary, fontsize=10)
    
    # Clauses (simplified)
    y += 220
    page.insert_text((50, y), "Risk Clauses:", fontsize=14, color=(0, 0, 1))
    y += 20
    clauses = (analytics.get("clause_risk") or {}).get("clauses", [])
    for c in clauses[:5]: # Show top 5
        risk = c.get("risk", "Unknown")
        clause_txt = c.get("clause", "Unknown clause")
        text = f"[{risk}] {clause_txt}"
        page.insert_text((50, y), text, fontsize=10)
        y += 15
        if y > 800:
            break
            
    pdf_bytes = doc.tobytes()
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{doc_id}.pdf"})
