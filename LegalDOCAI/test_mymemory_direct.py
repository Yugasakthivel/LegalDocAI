
import httpx
import json

def test_mymemory():
    text = "உரிமையாளர்கள் பெயர்"
    source = "ta"
    target = "en"
    params = {
        "q": text,
        "langpair": f"{source}|{target}"
    }
    print(f"Testing MyMemory with: '{text}' ({source}->{target})")
    try:
        r = httpx.get("https://api.mymemory.translated.net/get", params=params, timeout=15)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        if str(data.get("responseStatus")) == "200":
            translated = (data.get("responseData") or {}).get("translatedText")
            print(f"Translated: {translated}")
        else:
            print(f"Error: {data.get('responseDetails')}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_mymemory()
