from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.state.model_registry import ModelRegistry
from app.state.dataset_registry import DatasetRegistry


class EvaluationEngine:

    def evaluate_model(self, model_id: str) -> Dict[str, Any]:

        model_entry = ModelRegistry.get(model_id)
        model = model_entry["model"]
        metadata = model_entry["metadata"]

        dataset_id = metadata["dataset_id"]
        target_column = metadata["target_column"]
        features = metadata["features"]

        df = DatasetRegistry.get(dataset_id)

        X = df[features]
        y = df[target_column]

        predictions = model.predict(X)

        # Classification vs Regression detection
        if self._is_classification(y):
            return self._classification_metrics(y, predictions)

        return self._regression_metrics(y, predictions)

    def _is_classification(self, y):
        return not np.issubdtype(y.dtype, np.number)


    def _classification_metrics(self, y_true, y_pred):

        return {
            "task": "classification",
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        }

    def _regression_metrics(self, y_true, y_pred):

        return {
            "task": "regression",
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }
