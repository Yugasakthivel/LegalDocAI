from pymongo import MongoClient, ASCENDING
from .config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
documents_collection = db["documents"]

# Ensure simple index on doc_id (unique)
documents_collection.create_index([("doc_id", ASCENDING)], unique=True)
