import uuid
import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, GridSearchCV

from app.state.dataset_registry import DatasetRegistry
from app.state.model_registry import ModelRegistry


class MLEngine:

    # ==========================================================
    # MODEL FACTORY (LLM SAFE)
    # ==========================================================

    def _get_model(self, model_type: str, problem_type: str):

        model_type = model_type.lower().strip()
        problem_type = problem_type.lower().strip()

        # ---- Aliases to prevent LLM naming mismatch ----
        aliases = {
            "linear_regression": "linear",
            "linreg": "linear",
            "rf": "random_forest",
            "randomforest": "random_forest",
            "gboost": "gradient_boosting",
            "gb": "gradient_boosting",
            "logistic_regression": "logistic",
            "logreg": "logistic",
        }

        model_type = aliases.get(model_type, model_type)

        models = {

            "regression": {
                "linear": LinearRegression(),
                "random_forest": RandomForestRegressor(random_state=42),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
            },

            "classification": {
                "logistic": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(random_state=42),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
            }
        }

        if problem_type not in models:
            raise ValueError(f"Unsupported problem type '{problem_type}'")

        if model_type not in models[problem_type]:
            raise ValueError(
                f"Unsupported model type '{model_type}' for '{problem_type}'"
            )

        return models[problem_type][model_type]

    # ==========================================================
    # TRAIN MODEL
    # ==========================================================

    def train_model(
        self,
        model_type: str,
        dataset_id: str,
        target_column: str,
        problem_type: str = "regression"
    ) -> Dict[str, Any]:

        df = DatasetRegistry.get(dataset_id)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Safe baseline → numeric features only
        X = X.select_dtypes(include=["number"])

        if X.empty:
            raise ValueError("No numeric features available for training.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = self._get_model(model_type, problem_type)
        model.fit(X_train, y_train)

        model_id = str(uuid.uuid4())

        # 🔥 FIX: Correct ModelRegistry signature
        ModelRegistry.register(
            model_id,
            model,
            {
                "problem_type": problem_type,
                "dataset_id": dataset_id,
                "target": target_column,
                "features": list(X.columns),
                "model_type": model_type
            }
        )

        return {
            "model_id": model_id,
            "model_type": model_type,
            "problem_type": problem_type,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features_used": list(X.columns)
        }

    # ==========================================================
    # HYPERPARAMETER TUNING
    # ==========================================================

    def tune_model(
        self,
        model_type: str,
        dataset_id: str,
        target_column: str,
        problem_type: str = "regression"
    ) -> Dict[str, Any]:

        df = DatasetRegistry.get(dataset_id)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = X.select_dtypes(include=["number"])

        if X.empty:
            raise ValueError("No numeric features available for tuning.")

        base_model = self._get_model(model_type, problem_type)

        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
            }
        }

        # Normalize name for grid
        grid_key = model_type.lower().strip()
        grid_key = {
            "rf": "random_forest",
            "randomforest": "random_forest"
        }.get(grid_key, grid_key)

        if grid_key not in param_grids:
            raise ValueError("No tuning grid defined for this model")

        grid = GridSearchCV(
            base_model,
            param_grids[grid_key],
            cv=3,
            n_jobs=-1
        )

        grid.fit(X, y)

        best_model = grid.best_estimator_
        model_id = str(uuid.uuid4())

        ModelRegistry.register(
            model_id,
            best_model,
            {
                "problem_type": problem_type,
                "dataset_id": dataset_id,
                "target": target_column,
                "features": list(X.columns),
                "model_type": model_type,
                "best_params": grid.best_params_
            }
        )

        return {
            "model_id": model_id,
            "best_params": grid.best_params_
        }

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(
        self,
        model_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        model_entry = ModelRegistry.get(model_id)

        if model_entry is None:
            raise ValueError(f"Model '{model_id}' not found.")

        model = model_entry["model"]
        metadata = model_entry["metadata"]

        expected_features = metadata["features"]

        row = []
        for feature in expected_features:
            if feature not in input_data:
                raise ValueError(f"Missing feature '{feature}' in input data.")
            row.append(input_data[feature])

        X_input = np.array(row).reshape(1, -1)
        prediction = model.predict(X_input)[0]

        return {
            "model_id": model_id,
            "prediction": float(prediction)
        }
