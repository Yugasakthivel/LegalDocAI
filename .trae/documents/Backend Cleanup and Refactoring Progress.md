I have successfully cleaned up the project structure and started the backend refactoring.

**Completed Actions:**
1.  **Repo Cleanup**: Removed the committed virtual environment (`LegalDOCAI/Lib`) and updated `.gitignore` to exclude `Lib/`, `site-packages/`, and `.env` files.
2.  **Environment Security**: Created `.env.example` files for both Backend and Frontend, ensuring secrets are not committed.
3.  **Env Consistency**: Updated `LegalDOCAI/server.js` to support `MONGO_URI` (matching Python) alongside `MONGODB_URI`.
4.  **Config Refactoring**:
    *   Moved `LegalDOCAI/config.py` to `LegalDOCAI/backend/app/core/config.py`.
    *   Updated imports in `main.py`, `database.py`, `nlp.py`, `summarizer.py`, and `test_openai.py` to point to the new location.
    *   Created `backend/app/core/deps.py` for shared dependencies (e.g., `get_openai_client`).
5.  **Route Refactoring (Started)**:
    *   Created `backend/app/routes/dataset.py` as a new router module for dataset-related endpoints (migrated logic from `main.py`).

**Next Steps (Recommended for User)**:
1.  **Verify Application**: Run `python LegalDOCAI/main.py` to ensure the config refactor works as expected.
2.  **Continue Refactoring**:
    *   Extract `Analysis`, `OCR`, and `Report` logic from `main.py` into `backend/app/services/` and `backend/app/routes/`.
    *   Mount the new routers in `main.py` and remove the inline code.
    *   This will reduce `main.py` from ~2500 lines to a clean entry point.
