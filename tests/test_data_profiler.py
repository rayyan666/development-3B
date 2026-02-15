import pandas as pd

from app.engines.data_profiler import DataProfiler

# Load dataset directly for testing
df = pd.read_csv("IndiaAI_BillingAndUsageDailyData.csv")

profiler = DataProfiler(df)
profile = profiler.profile()

import json
print(json.dumps(profile, indent=2))
