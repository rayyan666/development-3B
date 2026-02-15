import requests
import json
import re

from app.agent.tool_parser import ToolParser
from app.agent.conversation_memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.executor import PlanExecutor
from app.core.orchestrator import Orchestrator
from app.engines.data_profiler import DataProfiler
from app.engines.data_strategist_engine import DataStrategistEngine
from app.state.dataset_registry import DatasetRegistry


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ds-ml-assistant"


class ChatController:

    def __init__(self):
        self.memory = ConversationMemory()
        self.orchestrator = Orchestrator()
        self.planner = Planner()
        self.executor = PlanExecutor(self.orchestrator)
        self.pending_plan = None

    # --------------------------------------------------
    # LLM Query
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Extract prediction features
    # --------------------------------------------------

    def extract_features_from_text(self, text: str):
        pattern = r"([\w\. ]+)=([\d\.]+)"
        matches = re.findall(pattern, text)

        features = {}
        for key, value in matches:
            key = key.strip()
            if "." in value:
                features[key] = float(value)
            else:
                features[key] = int(value)

        return features

    # --------------------------------------------------
    # MAIN HANDLE
    # --------------------------------------------------

    def handle(self, user_message: str):

        lower_msg = user_message.lower()
        self.memory.add_user_message(user_message)
        self.memory.messages = self.memory.messages[-12:]

        # ==========================================================
        # STRATEGIST MODE
        # ==========================================================
        
        if "analyze strategy" in lower_msg:
        
            if not self.memory.last_dataset_id:
                return "No dataset loaded. Please load a dataset first."
        
            try:
                # Registry returns DataFrame directly
                df = DatasetRegistry.get(self.memory.last_dataset_id)
        
                profiler = DataProfiler(df)
                profile = profiler.profile()  # ✅ FIXED
        
                strategist = DataStrategistEngine(profile)
                strategy = strategist.generate_full_strategy()
        
                formatted = "\nDATA STRATEGY REPORT\n\n"
                formatted += f"Problem Type: {strategy.get('problem_type')}\n"
                formatted += f"Recommended Target: {strategy.get('recommended_target')}\n"
                formatted += f"Drop Columns: {strategy.get('drop_columns')}\n"
                formatted += f"Drop Due To Collinearity: {strategy.get('drop_due_to_collinearity')}\n"
                formatted += f"Risk Flags: {strategy.get('risk_flags')}\n\n"
        
                formatted += "LLM Strategic Reasoning:\n"
                formatted += strategy.get("llm_reasoning", "None")
        
                return formatted
        
            except Exception as e:
                return f"Strategist failed: {str(e)}"


        # ==========================================================
        # PLAN CONFIRMATION MODE
        # ==========================================================

        if self.pending_plan:

            if lower_msg in ["yes", "y"]:

                results = self.executor.execute(self.pending_plan["plan"])

                # 🔥 Update memory from executed steps
                for step in results:
                    tool = step.get("tool")
                    result_data = step.get("result", {})
                    actual_result = result_data.get("result", result_data)

                    if tool == "load_csv":
                        dataset_id = actual_result.get("dataset_id")
                        if dataset_id:
                            self.memory.last_dataset_id = dataset_id

                    if tool == "train_model":
                        model_id = actual_result.get("model_id")
                        if model_id:
                            self.memory.last_model_id = model_id

                self.pending_plan = None
                return json.dumps(results, indent=2)

            elif lower_msg in ["no", "cancel"]:
                self.pending_plan = None
                return "Plan cancelled."

            else:
                return "Please respond with 'yes' or 'no'."

        # ==========================================================
        # PLANNER MODE
        # ==========================================================

        try:
            plan_obj = self.planner.generate_plan(
                user_input=user_message,
                memory=self.memory
            )

            if "plan" in plan_obj:
                self.pending_plan = plan_obj

                formatted = "\nProposed Plan:\n\n"

                for i, step in enumerate(plan_obj["plan"], 1):
                    formatted += f"{i}. {step.get('tool')} → {step.get('parameters')}\n"

                formatted += f"\nExplanation: {plan_obj.get('explanation')}\n"
                formatted += "\nProceed? (yes/no)"

                return formatted

        except Exception as e:
            return f"Planner failed: {str(e)}"

        return "I could not understand the request."
