import os
import sys
from dotenv import load_dotenv
from google import genai

def list_models():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    
    if not api_key:
        print("No API Key found.")
        return

    print(f"Using Key Prefix: {api_key[:4]}...")
    
    try:
        client = genai.Client(api_key=api_key)
        # Verify valid models for generateContent
        print("Fetching available models...")
        # Note: The new SDK listing might differ, we'll try the standard way
        # If client.models.list() exists
        pager = client.models.list()
        
        print("\n--- Available Models ---")
        for model in pager:
            try:
                # Just print everything available
                print(f"- {model.name} (Display: {getattr(model, 'display_name', 'N/A')})")
            except Exception:
                print(f"- {model}")
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
