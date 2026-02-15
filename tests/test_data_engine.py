from app.engines.data_engine import DataEngine

engine = DataEngine()

engine.load_csv("IndiaAI_BillingAndUsageDailyData", "IndiaAI_BillingAndUsageDailyData.csv")

print(engine.preview("IndiaAI_BillingAndUsageDailyData"))
print(engine.run_sql("SELECT COUNT(*) FROM IndiaAI_BillingAndUsageDailyData"))
