import json


ALLOWED_TOOLS = {
    "load_csv",
    "preview",
    "run_sql",
    "train_model",
    "evaluate_model",
    "run_eda",
    "get_feature_importance",
}


class ToolParser:

    @staticmethod
    def extract_json(text: str):
        """
        Extract outermost JSON object from model output.
        Handles contamination before/after JSON.
        """
        if not text:
            return None

        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return None

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def normalize_parameters(params: dict):
        """
        Normalize LLM parameter inconsistencies.
        """
        normalized = {}

        for key, value in params.items():
            if isinstance(value, str):
                value = value.strip()

            key = key.lower()

            # Normalize file-related keys
            if key in ["file_name", "filename", "file_path", "filepath"]:
                normalized["path"] = value
                continue

            # Normalize dataset id naming
            if key in ["dataset", "dataset_name"]:
                normalized["dataset_id"] = value
                continue

            # Normalize model type casing
            if key == "model_type":
                normalized["model_type"] = value.lower()
                continue

            normalized[key] = value

        return normalized

    @staticmethod
    def validate_tool_call(data: dict):
        """
        Validate JSON structure and allowed tool names.
        """
        if not isinstance(data, dict):
            return None

        if "tool_call" not in data:
            return None

        tool_call = data["tool_call"]

        if not isinstance(tool_call, dict):
            return None

        tool_name = tool_call.get("name")

        if not tool_name:
            return None

        tool_name = tool_name.strip()

        if tool_name not in ALLOWED_TOOLS:
            return None

        parameters = tool_call.get("parameters", {})

        if not isinstance(parameters, dict):
            parameters = {}

        normalized_params = ToolParser.normalize_parameters(parameters)

        return {
            "name": tool_name,
            "parameters": normalized_params
        }

    @staticmethod
    def parse(text: str):
        """
        Full parse pipeline:
        Extract JSON → Validate → Normalize
        """
        json_data = ToolParser.extract_json(text)

        if not json_data:
            return None

        return ToolParser.validate_tool_call(json_data)
