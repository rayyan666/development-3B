import duckdb
import pandas as pd
from typing import Dict, Any
from app.state.dataset_registry import DatasetRegistry


class DataEngine:

    def __init__(self):
        self.con = duckdb.connect(database=':memory:')

    def load_csv(self, dataset_id: str, path: str) -> Dict[str, Any]:
        df = pd.read_csv(path)
        DatasetRegistry.register(dataset_id, df)

        # Register with DuckDB
        self.con.register(dataset_id, df)

        return {
            "status": "success",
            "tool": "load_csv",
            "result": {
                "dataset_id": dataset_id,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "preview_rows": df.head(10).to_dict(orient="records")
            }
        }


    def run_sql(self, query: str) -> Dict[str, Any]:
        result = self.con.execute(query).fetchdf()

        return {
            "rows": result.head(20).to_dict(orient="records"),
            "shape": result.shape
        }

    def preview(self, dataset_id: str, n: int = 5) -> Dict[str, Any]:
        df = DatasetRegistry.get(dataset_id)

        return {
            "preview": df.head(n).to_dict(orient="records"),
            "shape": df.shape
        }
