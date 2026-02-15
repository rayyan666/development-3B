import requests
import json
import re

from app.state.dataset_registry import DatasetRegistry

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ds-ml-assistant"


class Planner:

    def __init__(self):
        pass

    # --------------------------------------------------
    # Robust JSON extraction
    # --------------------------------------------------

    def extract_json(self, text: str):
        """
        Extract valid JSON from LLM output.
        Handles:
        - Markdown fences
        - Extra text before/after JSON
        - Partial formatting issues
        """

        text = text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"```.*?\n", "", text, flags=re.DOTALL)
            text = text.replace("```", "").strip()

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extracting outermost JSON block
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        raise ValueError("Planner could not extract valid JSON from model output.")

    # --------------------------------------------------
    # Build Dataset Schema Context (Safe Version)
    # --------------------------------------------------

    def build_dataset_context(self, memory):

        if memory.last_dataset_id is None:
            return ""

        dataset_entry = DatasetRegistry.get(memory.last_dataset_id)

        if dataset_entry is None:
            return ""

        # Handle both:
        # 1) registry storing dict {"data": df}
        # 2) registry storing dataframe directly
        if isinstance(dataset_entry, dict) and "data" in dataset_entry:
            df = dataset_entry["data"]
        else:
            df = dataset_entry

        if df is None:
            return ""

        columns = list(df.columns)
        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
        total_rows = df.shape[0]

        context = f"""
Dataset '{memory.last_dataset_id}' is currently loaded.

Dataset Schema:
Columns: {columns}
Numeric Columns: {numeric_columns}
Total Rows: {total_rows}
"""

        return context

    # --------------------------------------------------
    # Generate Plan
    # --------------------------------------------------

    def generate_plan(self, user_input: str, memory):

        dataset_context = self.build_dataset_context(memory)

        prompt = f"""
You are a machine learning planning agent.

Your goal is to generate a correct multi-step execution plan.

Available tools and required parameters:

1. load_csv(path: string, dataset_id: string)
2. run_eda(dataset_id: string)
3. train_model(model_type: string, dataset_id: string, target_column: string)
4. evaluate_model(model_id: string)
5. get_feature_importance(model_id: string)
6. predict(model_id: string, input_data: object)

Current memory:
Dataset: {memory.last_dataset_id}
Model: {memory.last_model_id}

{dataset_context}

Planning Rules:

- If dataset is already loaded, DO NOT call load_csv again.
- Always use correct dataset_id from memory.
- NEVER invent column names.
- If user asks to build a model:
    - Choose a reasonable numeric column as target.
    - Prefer regression if numeric.
    - Default model_type = "random_forest".
- If dataset is not loaded, load it first.
- Always include required parameters for each tool.
- If model exists, do not retrain unless user explicitly asks.

Return ONLY valid JSON in this exact format:

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
