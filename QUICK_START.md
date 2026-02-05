# Quick Start Guide - Unified Legal Analysis Dashboard

## 🚀 Getting Started

### **1. Backend Setup**

Ensure backend is running:
```bash
cd LegalDOCAI
python main.py
```

Expected output:
```
===============================================
🚀 LegalDocAI Backend Started
✅ Model: gpt-4o-mini
✅ Mongo: mongodb://localhost:27017
✅ DB: LegalDocAI_DB
✅ Tesseract: C:\Program Files\Tesseract-OCR\tesseract.exe
===============================================
INFO: Application startup complete.
```

### **1a. Backend Environment Variables**

Create a `.env` file in `LegalDOCAI/` (you can copy from `LegalDOCAI/.env.example`).

Required/important keys:
```
OPENAI_API_KEY=
MONGO_URI=mongodb://localhost:27017
DB_NAME=LegalDocAI_DB
TESSERACT_CMD=
```

### **1b. External Services Required**

- **MongoDB**
  - Default: `mongodb://localhost:27017`
  - If MongoDB is unreachable, the backend will fall back to local JSON files in the project working directory:
    - `local_db_documents.json`
    - `local_db_users.json`

- **Tesseract OCR**
  - Required for OCR on images/scanned PDFs.
  - Either set `TESSERACT_CMD` (Windows example: `C:\Program Files\Tesseract-OCR\tesseract.exe`) or ensure `tesseract` is in your PATH.

- **OpenAI**
  - Used for embeddings (RAG/search) and some AI analysis features.
  - If `OPENAI_API_KEY` is not configured, chat/RAG endpoints return `503` with a clear "OpenAI not configured" error.

### **2. Frontend Setup**

```bash
cd LegalDoc-FrontEnd
npm install
npm run local
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

### **3. Frontend Environment Variables**

Create `.env` in `LegalDoc-FrontEnd/`:
```
VITE_BACKEND_URL=http://localhost:8000
```

### **4. Dependencies & Common Mismatches**

- **Python**
  - This repo contains multiple `requirements.txt` files.
  - Use `LegalDOCAI/requirements.txt` as the canonical backend dependency set.

- **Node / Frontend**
  - Frontend dependencies are in `LegalDoc-FrontEnd/package.json`.
  - Use `npm run local` to start the dev server.

- **Git ignore caveat (backend JSON)**
  - `LegalDOCAI/.gitignore` currently ignores `*.json`, which can hide files like `LegalDOCAI/package.json`.
  - Recommended: remove or narrow the `*.json` ignore rule so `package.json` is not ignored.

- **Security note (OpenAI in frontend)**
  - The frontend has the `openai` npm package.
  - Do not expose `OPENAI_API_KEY` in the browser; keep OpenAI calls in the backend.

---

## 📖 Usage Flow

### **Step 1: Upload Document**
1. Navigate to http://localhost:5173/upload
2. Click "Upload Sample Employment Agreement" or select a file
3. Wait for processing checklist to complete
4. Auto-redirects to `/analysis?doc_id=...`

### **Step 2: View Analysis**
1. Dashboard loads with all 12 modules
2. Module status badges show progress (⏳ loading → ✅ success)
3. Risk banner displays overall risk level
4. Metrics tiles show key statistics

### **Step 3: Explore Tabs**
- **Overview**: Document type, processing time, verification status
- **Risks**: Clause-wise risks with explanations
- **Missing**: Missing mandatory clauses
- **Entities**: Names, organizations, emails, phones
- **Timeline**: Extracted dates and events
- **Chat**: Ask questions about the document
- **Comparison**: Compare with another document
- **Report**: Download PDF or JSON

### **Step 4: History**
1. Go to Dashboard
2. Click "History" in sidebar
3. Select a previous document
4. Dashboard loads instantly (no re-processing)
5. Click "Re-Analyze" to re-run pipeline

---

## 🔍 Key Components

### **DocumentContext** (`src/context/DocumentContext.jsx`)
- Central state management
- Manages all 12 modules' status and results
- Provides actions: setDocId, setModuleStatus, setModuleResult, hydrateState

### **UnifiedAnalysisDashboard** (`src/pages/UnifiedAnalysisDashboard.jsx`)
- Main page component
- Loads documents (new or from history)
- Displays all 12 modules with status badges
- Handles re-analysis

### **FileUpload** (`src/components/FileUpload.jsx`)
- Upload interface
- Processing checklist
- Integrates with DocumentContext

---

## 🧪 Testing Scenarios

### **Scenario 1: New Upload**
```
1. Go to /upload
2. Upload file
3. See processing checklist (5 steps)
4. Redirected to /analysis?doc_id=...
5. All 12 module badges show ✅ success
6. Risk banner displays
7. Metrics tiles populated
8. Tabs show data
```

### **Scenario 2: History Load**
```
1. Go to /dashboard
2. Click History
3. Select a previous document
4. Redirected to /analysis?doc_id=...
5. Dashboard loads INSTANTLY (no spinner)
6. State hydrated from saved outputs
7. No re-processing
```

### **Scenario 3: Re-Analyze**
```
1. On analysis page
2. Click "Re-Analyze" button
3. Confirmation dialog appears
4. Click "Confirm"
5. Module badges show ⏳ loading
6. Pipeline re-runs
7. Badges show ✅ success
8. UI refreshes with new results
```

### **Scenario 4: Error Handling**
```
1. Simulate network error (DevTools → Offline)
2. Try to upload
3. Error banner appears
4. UI still renders
5. Can retry via "Re-Analyze"
```

---

## 📊 Module Status Indicators

| Icon | Status | Meaning |
|------|--------|---------|
| ✅ | success | Module completed successfully |
| ⏳ | loading | Module is processing |
| ❌ | error | Module failed |
| ⭕ | idle | Module not started |

---

## 🔗 API Endpoints Used

### **Orchestration**
- `POST /api/full` → Runs all modules (M1-M12)

### **History**
- `GET /api/document/history` → List all documents

### **Reports**
- `GET /api/report/pdf?doc_id=...` → PDF download
- `GET /api/modules/report?doc_id=...` → JSON report

### **Chat**
- `POST /api/modules/rag/query` → Q&A

**Note**: If `OPENAI_API_KEY` is not configured, chat/RAG endpoints will return `503` with a clear "OpenAI not configured" error.

---

## 🐛 Troubleshooting

### **Issue: Dashboard shows "No document loaded"**
- Check URL has `?doc_id=...` parameter
- Verify backend is running
- Check browser console for errors

### **Issue: History not loading**
- Verify `GET /api/document/history` returns valid JSON
- Check MongoDB is running
- Ensure doc_id exists in database

### **Issue: Modules stuck on "loading"**
- Check backend logs for errors
- Verify OpenAI API key is set
- Check network tab in DevTools

### **Issue: "Re-Analyze" not working**
- Verify `POST /api/full` endpoint exists
- Check doc_id is valid
- Look for errors in browser console

---

## 📝 File Structure

```
LegalDoc-FrontEnd/
├── src/
│   ├── context/
│   │   ├── AuthContext.jsx
│   │   └── DocumentContext.jsx          ← NEW
│   ├── pages/
│   │   ├── UnifiedAnalysisDashboard.jsx ← NEW
│   │   ├── UploadPage.jsx
│   │   └── ...
│   ├── components/
│   │   ├── FileUpload.jsx               ← UPDATED
│   │   └── ...
│   └── App.jsx                          ← UPDATED
└── .env                                 ← ADD THIS
```

---

## 🎯 Next Steps

1. **Test the complete flow**
   - Upload → Analysis → History → Re-Analyze

2. **Verify all modules**
   - Check each tab displays correct data
   - Verify module status badges update

3. **Test error scenarios**
   - Network failure
   - Invalid doc_id
   - Missing data

4. **Performance testing**
   - Measure load time
   - Check memory usage
   - Monitor API response times

5. **Deploy to production**
   - Build frontend: `npm run build`
   - Deploy to hosting (Vercel, AWS, etc.)
   - Set production backend URL

---

## 📞 Support

For issues or questions:
1. Check browser console for errors
2. Check backend logs
3. Review `UNIFIED_DASHBOARD_ARCHITECTURE.md`
4. Review `IMPLEMENTATION_SUMMARY.md`

---

**Status**: ✅ Ready for testing and deployment
