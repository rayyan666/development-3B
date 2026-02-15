import pandas as pd
from typing import Dict, Any

from app.state.dataset_registry import DatasetRegistry


class EDAEngine:

    def run_eda(self, dataset_id: str) -> Dict[str, Any]:

        df = DatasetRegistry.get(dataset_id)

        if df is None:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        eda_result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values_total": int(df.isnull().sum().sum()),
            "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
        }

        # Correlation only if numeric columns exist
        numeric_df = df.select_dtypes(include=["number"])

        if not numeric_df.empty:
            eda_result["correlation_matrix"] = numeric_df.corr().to_dict()
        else:
            eda_result["correlation_matrix"] = {}

        return eda_result
