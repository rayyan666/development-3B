import requests
import json
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ds-ml-assistant"


class Planner:

    # --------------------------------------------------
    # Robust JSON Extraction
    # --------------------------------------------------

    def extract_json(self, text: str):

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        raise ValueError("Planner could not extract valid JSON.")

    # --------------------------------------------------
    # Generate Plan
    # --------------------------------------------------

    def generate_plan(self, user_input: str, memory):
    
        lower_input = user_input.lower()
    
        # If user wants to load dataset → allow planning
        if "load" in lower_input and ".csv" in lower_input:
            pass
        else:
            # For all other tasks, require dataset
            if not memory.last_dataset_id:
                raise ValueError("No dataset loaded.")

        prompt = f"""
You are a machine learning planning agent.

Current state:
Dataset Loaded: {memory.last_dataset_id}
Model Loaded: {memory.last_model_id}

Available tools:

1. load_csv(path: string, dataset_id: string)
2. run_eda(dataset_id: string)
3. analyze_strategy(dataset_id: string)
4. train_model(model_type: string, dataset_id: string, target_column: string, problem_type: string)
5. evaluate_model(model_id: string)
6. tune_model(model_type: string, dataset_id: string, target_column: string, problem_type: string)
7. predict(model_id: string, input_data: object)

Rules:

- NEVER invent dataset_id.
- Always use dataset_id = "{memory.last_dataset_id}" if dataset exists.
- If model exists, use model_id = "{memory.last_model_id}".
- Always include required parameters.
- Return ONLY valid JSON in this format:

{{
  "plan": [
    {{
      "tool": "tool_name",
      "parameters": {{}}
    }}
  ],
  "explanation": "short explanation"
}}

User request:
{user_input}
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        if response.status_code != 200:
            raise ValueError(f"Ollama error: {response.text}")

        raw_output = response.json()["response"].strip()

        return self.extract_json(raw_output)
