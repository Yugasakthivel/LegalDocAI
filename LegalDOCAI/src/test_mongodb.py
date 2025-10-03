from pymongo import MongoClient

# Connect to MongoDB
# For local MongoDB
client = MongoClient("mongodb://localhost:27017")

# For MongoDB Atlas, use your connection string instead
# client = MongoClient("mongodb+srv://<username>:<password>@cluster0.mongodb.net/test")

# Create/use database
db = client['LegalDocAI']

# Create/use collection
collection = db['documents']

# Insert a test document
test_doc = {
    "title": "Test Document",
    "content": "This is a sample document for testing LegalDocAI MongoDB integration."
}

inserted_id = collection.insert_one(test_doc).inserted_id
print("Inserted document ID:", inserted_id)

# Retrieve and print all documents
print("\nAll documents in collection:")
for doc in collection.find():
    print(doc)
