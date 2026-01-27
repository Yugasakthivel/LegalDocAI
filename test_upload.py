import requests
import json

url = "http://127.0.0.1:8000/api/upload"

print("Testing TXT file upload...")
with open("test_legal_doc.txt", "rb") as f:
    response = requests.post(url, files={"file": f})

print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))
