import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import cross_val_score

from app.state.dataset_registry import DatasetRegistry
from app.state.model_registry import ModelRegistry


class EvaluationEngine:

    # ==========================================================
    # EVALUATE MODEL
    # ==========================================================

    def evaluate_model(
        self,
        model_id: str
    ) -> Dict[str, Any]:

        model_entry = ModelRegistry.get(model_id)

        if model_entry is None:
            raise ValueError(f"Model '{model_id}' not found.")

        pipeline = model_entry["model"]
        metadata = model_entry["metadata"]

        dataset_id = metadata["dataset_id"]
        target_column = metadata["target"]
        problem_type = metadata["problem_type"]

        df = DatasetRegistry.get(dataset_id)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # =============================
        # Predictions
        # =============================

        predictions = pipeline.predict(X)

        # =============================
        # Regression
        # =============================

        if problem_type == "regression":

            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)

            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring="r2"
            )

            metrics = {
                "r2": float(r2),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std())
            }

        # =============================
        # Classification
        # =============================

        elif problem_type == "classification":

            acc = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, average="weighted", zero_division=0)
            recall = recall_score(y, predictions, average="weighted", zero_division=0)
            f1 = f1_score(y, predictions, average="weighted", zero_division=0)

            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring="accuracy"
            )

            metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std())
            }

        else:
            raise ValueError("Unsupported problem type")

        # Store metrics inside model registry
        metadata["evaluation"] = metrics

        return {
            "status": "success",
            "tool": "evaluate_model",
            "result": {
                "model_id": model_id,
                "problem_type": problem_type,
                "metrics": metrics
            }
        }
