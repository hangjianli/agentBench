import os
import litellm
from dotenv import load_dotenv

def test_connection():
    load_dotenv()
    
    api_key = os.getenv("LMSTUDIO_TOKEN", "lm-studio")
    api_base = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")
    
    print(f"Testing litellm connection...")
    print(f"Model: openai/{model}")
    print(f"API Base: {api_base}")
    
    # Enable LiteLLM debug logging
    litellm.set_verbose = True
    
    try:
        response = litellm.completion(
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_base=api_base,
            api_key=api_key
        )
        print("\n--- Success ---")
        print(response.choices[0].message.content)
    except Exception as e:
        print("\n--- Failure ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    test_connection()
