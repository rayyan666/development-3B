from typing import Dict, Any, List
import numpy as np


class DataStrategistEngine:

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile

        self.numeric_columns = profile.get("numeric_columns", [])
        self.categorical_columns = profile.get("categorical_columns", [])
        self.column_profile = profile.get("column_profile", {})
        self.multicollinearity = profile.get("multicollinearity", [])
        self.rows = profile.get("dataset_summary", {}).get("rows", 0)

    # --------------------------------------------------
    # Utility Filters
    # --------------------------------------------------

    def is_constant(self, col: str) -> bool:
        return self.column_profile[col].get("unique_values", 0) <= 1

    def looks_like_id(self, col: str) -> bool:
        return (
            "id" in col.lower()
            or col.lower().endswith("_id")
            or col.lower() == "s. no."
        )

    def low_variance_numeric(self, col: str) -> bool:
        std = self.column_profile[col].get("std")
        return std == 0 or std is None

    # --------------------------------------------------
    # Feature Cleaning
    # --------------------------------------------------

    def get_drop_columns(self) -> List[str]:

        drops = []

        for col in self.numeric_columns + self.categorical_columns:

            if self.is_constant(col):
                drops.append(col)
                continue

            if self.looks_like_id(col):
                drops.append(col)
                continue

            if col in self.numeric_columns and self.low_variance_numeric(col):
                drops.append(col)

        return list(set(drops))

    # --------------------------------------------------
    # Handle Multicollinearity
    # --------------------------------------------------

    def resolve_multicollinearity(self) -> List[str]:

        to_drop = []

        for pair in self.multicollinearity:
            if abs(pair.get("correlation", 0)) >= 0.95:
                # Drop second feature deterministically
                to_drop.append(pair["feature_2"])

        return list(set(to_drop))

    # --------------------------------------------------
    # Determine Target Candidates
    # --------------------------------------------------

    def determine_targets(self):

        valid_numeric = []

        for col in self.numeric_columns:
            if (
                not self.is_constant(col)
                and not self.looks_like_id(col)
            ):
                valid_numeric.append(col)

        # Simple rule:
        # If numeric column has > 5 unique values → regression candidate
        regression_targets = []
        classification_targets = []

        for col in valid_numeric:
            unique_vals = self.column_profile[col].get("unique_values", 0)

            if unique_vals > 5:
                regression_targets.append(col)
            elif 2 <= unique_vals <= 5:
                classification_targets.append(col)

        return regression_targets, classification_targets

    # --------------------------------------------------
    # Dataset Risk Detection
    # --------------------------------------------------

    def risk_flags(self):

        flags = []

        if self.rows < 200:
            flags.append("small_dataset")

        if len(self.multicollinearity) > 0:
            flags.append("high_collinearity")

        return flags

    # --------------------------------------------------
    # Final Strategy Output
    # --------------------------------------------------

    def build_strategy(self):

        drop_columns = self.get_drop_columns()
        collinear_drops = self.resolve_multicollinearity()
        regression_targets, classification_targets = self.determine_targets()
        risks = self.risk_flags()

        problem_type = None
        recommended_target = None

        if regression_targets:
            problem_type = "regression"
            recommended_target = regression_targets[0]
        elif classification_targets:
            problem_type = "classification"
            recommended_target = classification_targets[0]
        else:
            problem_type = "unsupervised"

        return {
            "problem_type": problem_type,
            "recommended_target": recommended_target,
            "drop_columns": drop_columns,
            "drop_due_to_collinearity": collinear_drops,
            "regression_targets": regression_targets,
            "classification_targets": classification_targets,
            "risk_flags": risks,
        }
