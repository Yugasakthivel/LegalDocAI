# Project Structure (Professional Baseline)

This repository currently contains three main app areas:

- `LegalDoc-FrontEnd/` → React + Vite frontend
- `LegalDOCAI/` → FastAPI backend and ML/NLP pipeline
- `legal_dataset/` → training and annotation dataset assets

## Recommended ownership

- Frontend UI/UX, routing, pages, components → `LegalDoc-FrontEnd/`
- API routes, processing services, OCR/NLP/ML logic → `LegalDOCAI/backend/app/`
- Sample/training data only (non-runtime) → `legal_dataset/`

## Runtime/generated paths (not source code)

These should stay untracked in git:

- `LegalDOCAI/uploads/`
- `LegalDOCAI/chroma/`
- `LegalDOCAI/node_modules/`
- root-level `uploads/`, `chroma/`
- local db snapshots (`local_db_documents.json`, `local_db_users.json`)

## CI baseline

CI workflow exists at:

- `.github/workflows/ci.yml`

It validates:

- Frontend production build
- Backend critical module compile smoke checks

## Next optional improvements

1. Add backend `pytest` suite and run in CI.
2. Add frontend lint (`eslint`) check in CI.
3. Add dependency/security scan (e.g., `pip-audit`, `npm audit --omit=dev`).
4. Move ad-hoc test scripts into a clear `tests/` layout.
