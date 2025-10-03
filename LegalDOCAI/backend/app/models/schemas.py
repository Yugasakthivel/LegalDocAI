# backend/app/models/schemas.py
from pydantic import BaseModel
from typing import List

class SearchResponse(BaseModel):
    results: List[dict]

class DocumentResponse(BaseModel):
    id: str
    file_name: str
    summary: str
