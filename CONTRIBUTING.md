# Contributing Guide

Thanks for contributing to LegalDocAI.

## 1) Branching

- Create feature/fix branches from `main`.
- Suggested naming:
  - `feature/<short-name>`
  - `fix/<short-name>`
  - `chore/<short-name>`

## 2) Local setup

### Frontend

```bash
npm --prefix LegalDoc-FrontEnd ci
npm --prefix LegalDoc-FrontEnd run local
```

### Backend

```bash
cd LegalDOCAI
python -m pip install -r requirements.txt
python main.py
```

## 3) Before opening PR

- Frontend build must pass:

```bash
npm --prefix LegalDoc-FrontEnd run build
```

- Backend smoke compile must pass:

```bash
python -m py_compile LegalDOCAI/main.py LegalDOCAI/backend/app/routes/__init__.py LegalDOCAI/backend/app/routes/ai.py
```

## 4) Commit hygiene

- Keep commits focused and small.
- Use clear messages, e.g.:
  - `fix: handle OCR fallback on empty text`
  - `chore: ignore runtime upload artifacts`

## 5) Runtime/generated files

Do not commit generated runtime data:

- `LegalDOCAI/uploads/`
- `LegalDOCAI/chroma/`
- `LegalDOCAI/node_modules/`
- root `uploads/`, `chroma/`

These are already ignored in `.gitignore`.
