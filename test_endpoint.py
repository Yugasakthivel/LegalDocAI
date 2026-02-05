import requests

url = "http://localhost:8000/api/analysis/advanced"
data = {"text": "This is a short test."}

try:
    print(f"Testing {url} with long timeout...")
    response = requests.post(url, data=data, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
