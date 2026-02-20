from app.engines.data_engine import DataEngine
from app.engines.ml_engine import MLEngine
from app.engines.evaluation_engine import EvaluationEngine
from app.engines.eda_engine import EDAEngine
from app.engines.explain_engine import ExplainEngine
from app.engines.data_strategist_engine import DataStrategistEngine


class InvalidTool(Exception):
    pass


class Dispatcher:

    def __init__(self):
        self.data_engine = DataEngine()
        self.ml_engine = MLEngine()
        self.eval_engine = EvaluationEngine()
        self.eda_engine = EDAEngine()
        self.explain_engine = ExplainEngine()
        self.strategist = DataStrategistEngine()

    def dispatch(self, tool_name: str, parameters: dict):

        if tool_name == "load_csv":
            return self.data_engine.load_csv(**parameters)

        elif tool_name == "run_eda":
            return self.eda_engine.run_eda(**parameters)

        elif tool_name == "get_feature_importance":
            return self.explain_engine.get_feature_importance(**parameters)

        elif tool_name == "train_model":
            return self.ml_engine.train_model(**parameters)

        elif tool_name == "evaluate_model":
            return self.eval_engine.evaluate_model(**parameters)

        elif tool_name == "predict":
            return self.ml_engine.predict(
                parameters["model_id"],
                parameters["input_data"]
            )
        
        elif tool_name == "tune_model":
            return self.ml_engine.tune_model(**parameters)


        else:
            raise InvalidTool(f"Unknown tool: {tool_name}")

