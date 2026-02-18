
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.app.core.deps import init_openai_from_env, get_openai_client, is_openai_ready
from backend.app.core.config import OPENAI_MODEL

def test_openai():
    print("Testing OpenAI initialization...")
    success = init_openai_from_env()
    print(f"init_openai_from_env() success: {success}")
    print(f"is_openai_ready(): {is_openai_ready()}")
    
    if not is_openai_ready():
        print("❌ OpenAI client is not ready. Check your .env file and OPENAI_API_KEY.")
        return

    try:
        client = get_openai_client()
        print(f"Using model: {OPENAI_MODEL}")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        print("✅ OpenAI API Call Success!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ OpenAI API Call Failed: {e}")

if __name__ == "__main__":
    test_openai()
