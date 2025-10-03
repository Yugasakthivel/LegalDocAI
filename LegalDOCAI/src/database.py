from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["legal_documents"]
collection = db["documents"]

def insert_document(doc):
    collection.insert_one(doc)

def find_documents(query):
    return list(collection.find(query))

if __name__ == "__main__":
    import os
    os.chdir(r"d:\Project\LegalDocAI\LegalDOCAI")
    import main_pipeline
def save_document(title, content, summary=None, file_path=None):
    """
    Save a processed document to MongoDB
    """
    doc = {
        "title": title,
        "content": content,
        "summary": summary,
        "file_path": file_path
    }
    result = collection.insert_one(doc)
    print(f"Document inserted with ID: {result.inserted_id}")
