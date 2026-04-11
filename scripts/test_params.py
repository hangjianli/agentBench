import os
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
os.environ["OPENAI_API_KEY"] = os.getenv("LMSTUDIO_TOKEN", "lm-studio")
os.environ["OPENAI_API_BASE"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
model_name = os.getenv(
    "AGENT_BACKBONE_MODEL",
    os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b"),
)
litellm_model = f"openai/{model_name}"

def test_inference_params(params: dict, label: str):
    print(f"\n--- Testing: {label} ---")
    print(f"Params: {params}")
    try:
        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": "Explain the concept of quantum entanglement in 2 sentences."}],
            **params
        )
        print(f"Response (first 100 chars): {response.choices[0].message.content[:100]}...")
        # Check for reasoning if applicable
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            print(f"Reasoning Found: {response.choices[0].message.reasoning_content[:100]}...")
        else:
            print("No reasoning_content field found in response.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test 1: Standard Sampling
    test_inference_params({
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 50
    }, "Standard Sampling")

    # Test 2: Top K (Gemma specific)
    # Note: LiteLLM might need to pass this through extra_headers or specific provider args if not standard
    test_inference_params({
        "temperature": 0.7,
        "extra_body": {"top_k": 50},
        "max_tokens": 50
    }, "Top-K Sampling")

    # Test 3: Thinking/Reasoning Mode
    # Some local backends use 'include_reasoning' or similar flags in the body
    test_inference_params({
        "extra_body": {"include_reasoning": True},
        "max_tokens": 200
    }, "Thinking Mode (Reasoning)")
