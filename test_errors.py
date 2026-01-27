import requests
import json

url = "http://127.0.0.1:8000/api/upload"

print("Testing invalid file type (.exe)...")
with open("test_invalid.exe", "rb") as f:
    response = requests.post(url, files={"file": ("test.exe", f)})

print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))
print()
