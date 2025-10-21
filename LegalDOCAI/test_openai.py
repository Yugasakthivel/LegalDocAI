# test_openai.py — Test OpenAI API Key
from config import client, OPENAI_MODEL

try:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0,
        max_tokens=5
    )
    print("✅ OpenAI Key works! Response:", response.choices[0].message.content)
except Exception as e:
    print("⚠️ OpenAI Key failed:", e)




