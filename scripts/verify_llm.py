import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp.decision_engine import DecisionEngine

def test_api():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    
    print(f"Checking API Key: {'Found' if api_key else 'Missing'}")
    if not api_key:
        print("Error: Please set LLM_API_KEY in your .env file.")
        return

    print("Initializing Decision Engine...")
    engine = DecisionEngine()
    
    context = {
        'drift_status': 'None',
        'current_metrics': {'accuracy': 0.95},
        'policies': {'threshold': 0.05}
    }
    
    print("Sending test request to LLM...")
    try:
        recommendation = engine.get_recommendation(context)
        print("\n--- Success! ---")
        print(f"LLM Response: {recommendation}")
    except Exception as e:
        print("\n--- Failure ---")
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    test_api()
