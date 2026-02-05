# Unified Legal Analysis Dashboard - Architecture & Implementation Guide

## Overview

The **Unified Legal Analysis Dashboard** is a production-ready, module-driven legal document analysis system that integrates all 12 backend modules into a single-page React application with proper state management, history rehydration, and graceful error handling.

---

## Architecture

### 1. **Central State Management (DocumentContext)**

**File**: `src/context/DocumentContext.jsx`

The entire application state is managed through a single Context + Reducer pattern:

```javascript
{
  doc_id: "UUID",
  file_info: {
    filename: "",
    type: "",
    pages: 0,
    ocr_lang: "",
    auth_score: 0,
  },
  module_status: {
    M1: "idle|loading|success|error",
    M2: "...",
    // ... M3-M12
  },
  results: {
    M1: null, // Ingest output
    M2: null, // OCR output
    // ... M3-M12
  },
  analytics: {
    legality_score: 0,
    combined_risk_index: 0,
    // ... other aggregated metrics
  },
  error: null,
  isLoading: false,
}
```

**Key Actions**:
- `SET_DOC_ID`: Set the document ID
- `SET_MODULE_STATUS`: Update module status (idle/loading/success/error)
- `SET_MODULE_RESULT`: Store module output
- `SET_ANALYTICS`: Update aggregated analytics
- `HYDRATE_STATE`: Restore entire state from history
- `RESET_STATE`: Clear all state

---

### 2. **Module Pipeline (M1 → M12)**

| Module | Endpoint | Purpose | UI Section |
|--------|----------|---------|-----------|
| M1 | POST /api/modules/ingest | Document upload & authenticity | Upload card |
| M2 | POST /api/modules/ocr | OCR & layout extraction | Page count badge |
| M3 | POST /api/modules/lang | Language detection & translation | Language badge |
| M4 | POST /api/modules/classify | Document type classification | Overview tab |
| M5 | POST /api/modules/ner | Named entity recognition | Entities tab |
| M6 | POST /api/modules/clauses | Clause presence detection | Missing clauses tab |
| M7 | POST /api/modules/risk | Risk detection & scoring | Risks tab |
| M8 | POST /api/modules/explain | Explainability & confidence | Overview tab |
| M9 | POST /api/modules/timeline | Timeline extraction | Timeline tab |
| M10 | POST /api/modules/rag/query | RAG-based Q&A | Chat tab |
| M11 | POST /api/modules/compare | Document comparison | Comparison tab |
| M12 | POST /api/modules/report | Report generation | Report tab |

---

### 3. **Data Flow**

#### **New Document Upload**
```
1. User uploads file via FileUpload component
2. POST /api/modules/ingest → get doc_id
3. FileUpload calls setDocId() and setFileInfo()
4. Redirect to /analysis?doc_id=...
5. UnifiedAnalysisDashboard loads and calls POST /api/full
6. Pipeline runs M1→M12, updates state at each step
7. State persisted to MongoDB via backend
```

#### **History Document Load**
```
1. User navigates to /analysis?doc_id=...
2. UnifiedAnalysisDashboard useEffect triggers
3. Fetch GET /api/document/history
4. Find doc_id in history
5. If analytics exist: hydrateState() from saved outputs (NO re-processing)
6. If missing: run POST /api/full to compute
7. Display UI from hydrated state
```

#### **Re-Analyze**
```
1. User clicks "Re-Analyze" button
2. Confirmation dialog appears
3. On confirm: POST /api/full with existing doc_id
4. Pipeline re-runs, updates state
5. UI refreshes with new results
```

---

### 4. **Component Structure**

#### **UnifiedAnalysisDashboard.jsx**
Main page component with:
- Header with doc_id and action buttons
- Risk banner (if risk > 0)
- Metrics tiles (Total Risks, Missing Clauses, Pages, Confidence)
- Module status panel (12 status badges)
- Tabbed interface (Overview, Risks, Missing, Entities, Timeline, Chat, Comparison, Report)
- Footer with navigation

#### **DocumentContext.jsx**
Global state provider with:
- Reducer for state mutations
- Action creators (setDocId, setModuleStatus, etc.)
- Provider component wrapping entire app

#### **FileUpload.jsx** (Updated)
Upload component with:
- Integration with DocumentContext
- Processing checklist (5 steps)
- Redirect to /analysis after upload

---

### 5. **History Rehydration Flow**

**Problem**: Clicking a history item used to re-run the entire pipeline.

**Solution**: 
1. Check if document exists in history with saved analytics
2. If yes: `hydrateState()` from stored outputs (instant load, no API calls)
3. If no: Run `POST /api/full` to compute (fallback)
4. "Re-Analyze" button explicitly triggers re-processing

**Code**:
```javascript
// In UnifiedAnalysisDashboard.jsx useEffect
const existing = history.find((d) => d?.doc_id === targetDocId);
if (existing && existing.analytics && Object.keys(existing.analytics).length > 0) {
  hydrateState({
    doc_id: existing.doc_id,
    file_info: existing.file_info || {},
    analytics: existing.analytics || {},
    results: { M1: ..., M2: ..., ... },
    module_status: { M1: "success", M2: "success", ... },
  });
  return; // No re-processing
}
```

---

### 6. **Error Handling & Graceful Degradation**

- Each module has a status badge (✅ success, ⏳ loading, ❌ error, ⭕ idle)
- If a module fails, its status = "error" but UI still renders
- Error banner at top shows last error
- Retry button on "Re-Analyze" to re-run pipeline
- AbortController prevents unhandled promise rejections on unmount

---

### 7. **Module Verification**

Each module section includes:
- **Status badge**: Visual indicator (✅/⏳/❌/⭕)
- **Color coding**: Green (success), Blue (loading), Red (error), Gray (idle)
- **Partial rendering**: If a module fails, show available data + retry option
- **Validation**: Check required fields before calling next module

---

## API Endpoints Used

### **Orchestration** (Fast path)
- `POST /api/full` → Runs M1→M12, returns aggregated analytics + results

### **Granular Modules** (Optional)
- `POST /api/modules/ingest` → M1 (upload)
- `POST /api/modules/classify` → M4 (classification)
- `POST /api/modules/ner` → M5 (entities)
- `POST /api/modules/risk` → M7 (risk)
- `POST /api/modules/timeline` → M9 (timeline)
- `POST /api/modules/rag/query` → M10 (chat)

### **History & Persistence**
- `GET /api/document/history` → List all documents with saved outputs
- `GET /document/{doc_id}` → Fetch single document (if exposed)

### **Reports**
- `GET /api/report/pdf?doc_id=...` → PDF download
- `GET /api/modules/report?doc_id=...` → JSON report

---

## Usage

### **1. Wrap App with DocumentProvider**

**App.jsx**:
```javascript
import { DocumentProvider } from './context/DocumentContext';

export default function App() {
  return (
    <AuthProvider>
      <DocumentProvider>
        <Router>
          <AppRoutes />
        </Router>
      </DocumentProvider>
    </AuthProvider>
  );
}
```

### **2. Use Context in Components**

**FileUpload.jsx**:
```javascript
import { DocumentContext } from "../context/DocumentContext";

export default function FileUpload() {
  const { setDocId, setFileInfo } = useContext(DocumentContext);
  
  // After upload:
  setDocId(json.doc_id);
  setFileInfo({ filename: file.name, ... });
}
```

**UnifiedAnalysisDashboard.jsx**:
```javascript
const { state, setModuleStatus, setModuleResult, hydrateState } = useContext(DocumentContext);

// Load from history:
hydrateState({ doc_id, file_info, analytics, results, module_status });

// Update module:
setModuleStatus("M5", "loading");
setModuleResult("M5", { entities: [...], stats: {...} });
```

### **3. Navigate to Dashboard**

After upload, user is redirected to:
```
/analysis?doc_id=<UUID>
```

UnifiedAnalysisDashboard loads and:
1. Checks history for doc_id
2. Hydrates state if found
3. Otherwise runs pipeline
4. Displays all 12 modules

---

## File Structure

```
src/
├── context/
│   ├── AuthContext.jsx
│   └── DocumentContext.jsx          ← NEW: Central state
├── pages/
│   ├── UnifiedAnalysisDashboard.jsx ← NEW: Main dashboard
│   ├── UploadPage.jsx
│   ├── ResultsPage.jsx
│   └── ...
├── components/
│   ├── FileUpload.jsx               ← UPDATED: Uses DocumentContext
│   ├── ChatBox.jsx
│   └── ...
└── App.jsx                          ← UPDATED: Wraps with DocumentProvider
```

---

## Key Features

✅ **Single-page unified dashboard**  
✅ **All 12 modules integrated with status tracking**  
✅ **History rehydration (no re-upload for old docs)**  
✅ **Re-Analyze button for explicit re-processing**  
✅ **Graceful error handling (partial rendering)**  
✅ **Module verification badges**  
✅ **Central state management (Context + Reducer)**  
✅ **AbortController for cleanup**  
✅ **Responsive UI with tabs**  
✅ **Download reports (PDF/JSON)**  

---

## Testing Checklist

- [ ] Upload new document → redirects to /analysis with doc_id
- [ ] Processing checklist shows 5 steps
- [ ] All 12 module badges show status
- [ ] Risk banner displays if risk > 0
- [ ] Metrics tiles show correct values
- [ ] Tabs (Overview, Risks, Missing, Entities, Timeline, Chat, Comparison, Report) work
- [ ] Click history item → loads without re-processing
- [ ] Click "Re-Analyze" → re-runs pipeline
- [ ] Error handling: if module fails, UI still renders with error badge
- [ ] Download PDF/JSON reports work
- [ ] Back to Dashboard / Upload New buttons work

---

## Future Enhancements

1. **Module-level retry**: Add retry button for individual failed modules
2. **Comparison UI**: Implement document comparison with side-by-side diff
3. **Export state**: Save/load analysis state as JSON
4. **Batch processing**: Analyze multiple documents
5. **Advanced filtering**: Filter risks by severity, type, etc.
6. **Audit trail**: Track all changes and re-analyses
7. **Collaboration**: Share analysis with team members

---

## Troubleshooting

### **History not loading**
- Check `GET /api/document/history` returns valid JSON
- Verify doc_id exists in history
- Check browser console for errors

### **Modules not updating**
- Verify `POST /api/full` returns correct structure
- Check module_status updates in Redux DevTools
- Ensure AbortController is not aborting requests

### **State not persisting**
- Verify backend saves to MongoDB
- Check `save_document()` in backend
- Ensure doc_id is consistent across requests

---

## Summary

The **Unified Legal Analysis Dashboard** provides a production-ready, modular, and maintainable solution for legal document analysis. It integrates all 12 backend modules, manages state centrally, handles history correctly, and provides a seamless user experience with proper error handling and verification.

**Key Principle**: Backend is the source of truth; frontend renders and manages UI state.
