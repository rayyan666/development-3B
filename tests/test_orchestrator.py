from app.core.orchestrator import Orchestrator

orch = Orchestrator()

orch.handle("load_csv", {"dataset_id": "aws", "path": "IndiaAI_BillingAndUsageDailyData.csv"})
train = orch.handle("train_model", {
    "model_type": "random_forest",
    "dataset_id": "aws",
    "target_column": "DailyHours"
})

model_id = train["result"]["model_id"]

print(orch.handle("evaluate_model", {"model_id": model_id}))

