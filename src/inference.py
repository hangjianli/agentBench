import os
from typing import List, Dict, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class InferenceClient:
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        api_key: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.api_key = api_key or os.getenv("LMSTUDIO_TOKEN", "lm-studio")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Sends a chat completion request to the inference server.
        Model precedence: Argument > .env (LMSTUDIO_MODEL) > First available from server.
        """
        if model is None:
            model = os.getenv("LMSTUDIO_MODEL")
            
        if model is None:
            models = self.client.models.list()
            if not models.data:
                raise RuntimeError("No models found on the inference server.")
            model = models.data[0].id
            print(f"No model specified. Using first available from server: {model}")

        print(f"Activated model: {model}")

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Quick smoke test
    client = InferenceClient()
    try:
        models = client.client.models.list()
        print("Available models:")
        for m in models.data:
            print(f" - {m.id}")
            
        test_msg = [{"role": "user", "content": "Hello, can you hear me?"}]
        response = client.chat_completion(test_msg)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"Error connecting to backend: {e}")
