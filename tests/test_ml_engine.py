from app.engines.ml_engine import MLEngine
from app.engines.data_engine import DataEngine

engine = DataEngine()
ml = MLEngine()

engine.load_csv("aws_data", "IndiaAI_BillingAndUsageDailyData.csv")

result = ml.train_model(
    model_type="random_forest",
    dataset_id="aws_data",
    target_column="DailyHours"
)

print(result)
