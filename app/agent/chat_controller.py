import requests
import json
import re

from app.agent.tool_parser import ToolParser
from app.agent.conversation_memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.executor import PlanExecutor
from app.core.orchestrator import Orchestrator


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ds-ml-assistant"


class ChatController:

    def __init__(self):
        self.memory = ConversationMemory()
        self.orchestrator = Orchestrator()
        self.planner = Planner()
        self.executor = PlanExecutor(self.orchestrator)
        self.pending_plan = None

    # ==========================================================
    # LLM Query
    # ==========================================================

    def query_ollama(self, prompt: str):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        if response.status_code != 200:
            return f"Error communicating with Ollama: {response.text}"

        return response.json()["response"].strip()

    # ==========================================================
    # Recursive Extractor (Fix Nested Results)
    # ==========================================================

    def _recursive_find(self, obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = self._recursive_find(v, key)
                if found:
                    return found
        return None

    # ==========================================================
    # Plan Formatting
    # ==========================================================

    def format_plan(self, plan_obj):
        steps = plan_obj.get("plan", [])
        explanation = plan_obj.get("explanation", "")

        formatted = "\nProposed Plan:\n\n"

        for i, step in enumerate(steps, 1):
            formatted += f"{i}. {step.get('tool')} → {step.get('parameters')}\n"

        formatted += f"\nExplanation: {explanation}\n"
        formatted += "\nProceed? (yes/no)"

        return formatted

    # ==========================================================
    # Strategy Formatting
    # ==========================================================

    def format_strategy(self, strategy_result):

        result = strategy_result.get("result", strategy_result)

        summary = "\nDATASET SUMMARY\n\n"
        summary += f"Rows: {result.get('rows')}\n"
        summary += f"Columns: {result.get('columns')}\n\n"

        summary += "Numeric Columns:\n"
        summary += f"{result.get('numeric_columns')}\n\n"

        summary += "Categorical Columns:\n"
        summary += f"{result.get('categorical_columns')}\n\n"

        summary += (
            "Please define:\n"
            "- Problem type (regression / classification / time-series)\n"
            "- Target column (example: target = DailyHours)\n"
        )

        return summary

    # ==========================================================
    # Result Summarization
    # ==========================================================

    def summarize_results(self, results):
        return json.dumps(results, indent=2)

    # ==========================================================
    # MAIN HANDLE FUNCTION
    # ==========================================================

    def handle(self, user_message: str):

        lower_msg = user_message.lower()

        self.memory.add_user_message(user_message)
        self.memory.messages = self.memory.messages[-12:]

        # ------------------------------------------------------
        # CONFIRM PLAN MODE
        # ------------------------------------------------------

        if self.pending_plan:

            if lower_msg in ["yes", "y"]:

                results = self.executor.execute(self.pending_plan["plan"])
                self.pending_plan = None

                # Update memory properly (FIXED)
                for step in results:

                    tool = step.get("tool")
                    result = step.get("result", {})

                    # ---- Dataset Tracking ----
                    if tool == "load_csv":
                        dataset_id = self._recursive_find(result, "dataset_id")
                        if dataset_id:
                            self.memory.last_dataset_id = dataset_id

                    # ---- Model Tracking (FIXED) ----
                    if tool == "train_model":
                        model_id = self._recursive_find(result, "model_id")
                        if model_id:
                            self.memory.last_model_id = model_id

                    # ---- Strategy Formatting ----
                    if tool == "analyze_strategy":
                        return self.format_strategy(result)

                return self.summarize_results(results)

            elif lower_msg in ["no", "cancel"]:
                self.pending_plan = None
                return "Plan cancelled."

            else:
                return "Please respond with 'yes' to proceed or 'no' to cancel."

        # ------------------------------------------------------
        # PLANNING MODE
        # ------------------------------------------------------

        try:
            plan_obj = self.planner.generate_plan(
                user_input=user_message,
                memory=self.memory
            )

            if "plan" in plan_obj:
                self.pending_plan = plan_obj
                return self.format_plan(plan_obj)

        except Exception as e:
            return f"Planner failed: {str(e)}"

        # ------------------------------------------------------
        # FALLBACK MODE
        # ------------------------------------------------------

        prompt = self.memory.build_prompt()
        model_output = self.query_ollama(prompt)
        tool_call = ToolParser.parse(model_output)

        if not tool_call:
            self.memory.add_assistant_message(model_output)
            return model_output

        tool_name = tool_call["name"]
        parameters = tool_call["parameters"]

        # Inject dataset automatically
        if tool_name in ["run_eda", "train_model", "analyze_strategy"]:
            if self.memory.last_dataset_id:
                parameters["dataset_id"] = self.memory.last_dataset_id

        # Inject model automatically
        if tool_name in ["evaluate_model", "get_feature_importance", "predict"]:
            if self.memory.last_model_id:
                parameters["model_id"] = self.memory.last_model_id

        result = self.orchestrator.handle(tool_name, parameters)

        return json.dumps(result, indent=2)
