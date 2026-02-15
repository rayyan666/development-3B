from app.engines.ml_engine import MLEngine
from app.engines.data_engine import DataEngine
from app.engines.evaluation_engine import EvaluationEngine

engine = DataEngine()
ml = MLEngine()
eval_engine = EvaluationEngine()

engine.load_csv("aws_data", "IndiaAI_BillingAndUsageDailyData.csv")

train_result = ml.train_model(
    model_type="random_forest",
    dataset_id="aws_data",
    target_column="DailyHours"
)

metrics = eval_engine.evaluate_model(train_result["model_id"])

print(metrics)
