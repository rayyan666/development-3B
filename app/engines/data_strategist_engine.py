from app.state.dataset_registry import DatasetRegistry
from app.engines.data_profiler import DataProfiler
import pandas as pd


class DataStrategistEngine:

    def analyze_strategy(self, dataset_id: str):

        df = DatasetRegistry.get(dataset_id)

        profiler = DataProfiler(df)
        profile = profiler.profile()

        numeric_cols = profile["numeric_columns"]
        multicollinearity = profile["multicollinearity"]
        warnings = []

        # ----------------------------
        # TARGET SELECTION
        # ----------------------------

        candidate_targets = []

        for col in numeric_cols:

            unique_values = profile["column_profile"][col]["unique_values"]
            std = profile["column_profile"][col].get("std", 0)

            # Ignore constants
            if unique_values <= 1:
                continue

            # Ignore ID-like columns
            if any(keyword in col.lower() for keyword in ["id", "no"]):
                continue

            # Must have variance
            if std == 0:
                continue

            candidate_targets.append(col)

        if not candidate_targets:
            return {
                "problem_type": None,
                "recommended_target": None,
                "drop_columns": None,
                "drop_due_to_collinearity": None,
                "risk_flags": ["no_valid_target_found"]
            }

        # Prefer highest variance column
        recommended_target = max(
            candidate_targets,
            key=lambda c: profile["column_profile"][c].get("std", 0)
        )

        problem_type = "regression"

        # ----------------------------
        # COLLINEARITY HANDLING
        # ----------------------------

        drop_due_to_collinearity = []

        for pair in multicollinearity:
            if pair["feature_1"] == recommended_target:
                drop_due_to_collinearity.append(pair["feature_2"])
            elif pair["feature_2"] == recommended_target:
                drop_due_to_collinearity.append(pair["feature_1"])

        # ----------------------------
        # DROP ID / LOW VALUE COLUMNS
        # ----------------------------

        drop_columns = []

        for col, meta in profile["column_profile"].items():

            if meta["unique_values"] == 1:
                drop_columns.append(col)

            if any(keyword in col.lower() for keyword in ["id", "name"]):
                drop_columns.append(col)

        # ----------------------------
        # RISK FLAGS
        # ----------------------------

        if profile["dataset_summary"]["rows"] < 100:
            warnings.append("small_dataset")

        if multicollinearity:
            warnings.append("high_collinearity")

        return {
            "problem_type": problem_type,
            "recommended_target": recommended_target,
            "drop_columns": list(set(drop_columns)),
            "drop_due_to_collinearity": list(set(drop_due_to_collinearity)),
            "risk_flags": warnings
        }
