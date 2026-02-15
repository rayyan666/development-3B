from typing import Dict, Any

from app.state.model_registry import ModelRegistry


class ExplainEngine:

    def get_feature_importance(self, model_id: str) -> Dict[str, Any]:

        model_entry = ModelRegistry.get(model_id)

        if model_entry is None:
            raise ValueError(f"Model '{model_id}' not found.")

        model = model_entry["model"]
        metadata = model_entry["metadata"]

        features = metadata["features"]

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            importance_dict = dict(
                sorted(
                    zip(features, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

            return {
                "model_id": model_id,
                "method": "feature_importances_",
                "feature_importance": importance_dict
            }

        # Linear models
        if hasattr(model, "coef_"):
            coefs = model.coef_

            # Handle multi-dimensional coef (classification)
            if hasattr(coefs, "__len__") and len(coefs.shape) > 1:
                coefs = coefs[0]

            importance_dict = dict(
                sorted(
                    zip(features, coefs),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            )

            return {
                "model_id": model_id,
                "method": "coef_",
                "feature_importance": importance_dict
            }

        raise ValueError("Model type does not support feature importance.")
