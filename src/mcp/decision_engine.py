from google import genai
import json
import os

class DecisionEngine:
    def __init__(self, api_key: str = None, model_name: str = None):
        api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        
        if not api_key:
            print("Warning: LLM_API_KEY not found. Decision Engine will be disabled.")
            self.client = None
        else:
            # Force 'gemini' mode (Generative Language API) by setting api_key
            # If using Vertex AI, one would use vertexai=True or project=...
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Warning: Client init failed: {e}")
                self.client = None

    def get_recommendation(self, context: dict) -> dict:
        """
        Uses LLM to decide on actions (retrain, rollback, etc.)
        Expects context with:
        - current_metrics
        - drift_status
        - policies
        """
        if not self.client:
            return {"action": "UNKNOWN", "reason": "No LLM Client"}

        prompt = self._construct_prompt(context)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            # Simple extraction strategy - expect JSON in text
            text = response.text
            # Clean md blocks if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"LLM Decision failed: {e}")
            return {"action": "ERROR", "reason": str(e)}

    def _construct_prompt(self, context: dict) -> str:
        return f"""
        You are an AI AutoML Orchestrator. detailed ML ops data is provided below.
        Analyze the situation and recommend an action: [RETRAIN, ROLLBACK, KEEP, PROMOTION].
        
        Context:
        - Drift Detected: {context.get('drift_status', 'Unknown')}
        - Current Model Metrics: {json.dumps(context.get('current_metrics', {}))}
        - Recent Changes: {context.get('recent_changes', 'None')}
        - MCP Policies: {json.dumps(context.get('policies', {}))}
        
        Output strictly valid JSON with keys: "action" (str), "reason" (str short summary), "confidence" (float 0-1).
        """
