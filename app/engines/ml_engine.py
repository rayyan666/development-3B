import uuid
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from app.state.dataset_registry import DatasetRegistry
from app.state.model_registry import ModelRegistry


class InvalidModelType(Exception):
    pass


class MLEngine:

    def train_model(
        self,
        model_type: str,
        dataset_id: str,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:

        df = DatasetRegistry.get(dataset_id)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Only keep numeric columns for MVP
        X = X.select_dtypes(include=["number"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = self._initialize_model(model_type, y)

        model.fit(X_train, y_train)

        model_id = str(uuid.uuid4())

        ModelRegistry.register(
            model_id,
            model,
            {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "model_type": model_type,
                "features": list(X.columns),
                "test_size": test_size
            }
        )

        return {
            "model_id": model_id,
            "model_type": model_type,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features_used": list(X.columns)
        }

    def _initialize_model(self, model_type: str, y: pd.Series):

        if model_type == "random_forest":

            if y.dtype.kind in "ifu":  # numeric → regression
                return RandomForestRegressor()
            else:
                return RandomForestClassifier()

        elif model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000)

        elif model_type == "linear_regression":
            return LinearRegression()

        else:
            raise InvalidModelType(f"Unsupported model type: {model_type}")
        
    def predict(self, model_id: str, input_data: dict):

        model_entry = ModelRegistry.get(model_id)
    
        if model_entry is None:
            raise ValueError(f"Model '{model_id}' not found.")
    
        model = model_entry["model"]
        metadata = model_entry["metadata"]
    
        features = metadata["features"]
    
        # Ensure correct order
        row = [input_data.get(feature) for feature in features]
    
        df = pd.DataFrame([row], columns=features)
    
        prediction = model.predict(df)[0]
    
        return {
            "model_id": model_id,
            "prediction": float(prediction)
        }
