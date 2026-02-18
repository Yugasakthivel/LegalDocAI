import requests
doc_id = 'd8d1cf1a-0ebd-48a3-80ac-3ca191fc6ce9'
url = f'http://127.0.0.1:8000/api/modules/report?doc_id={doc_id}'
try:
    r = requests.get(url)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}...")
except Exception as e:
    print(f"Error: {e}")
