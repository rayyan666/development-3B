import uuid
import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from app.state.dataset_registry import DatasetRegistry
from app.state.model_registry import ModelRegistry


class MLEngine:

    # ==========================================================
    # MODEL FACTORY
    # ==========================================================

    def _get_model(self, model_type: str, problem_type: str):

        model_type = model_type.lower()

        models = {

            "regression": {
                "linear": LinearRegression(),
                "linear_regression": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "ridge_regression": Ridge(alpha=1.0),
                "lasso": Lasso(alpha=0.1),
                "lasso_regression": Lasso(alpha=0.1),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "tree": DecisionTreeRegressor(random_state=42),
                "knn": KNeighborsRegressor(n_neighbors=5),
                "k_neighbors": KNeighborsRegressor(n_neighbors=5),
                "svr": SVR(kernel="rbf"),
                "support_vector": SVR(kernel="rbf"),
                "random_forest": RandomForestRegressor(random_state=42, n_estimators=100),
                "gradient_boosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
                "adaboost": AdaBoostRegressor(random_state=42, n_estimators=100),
                "extra_trees": ExtraTreesRegressor(random_state=42, n_estimators=100),
            },

            "classification": {
                "logistic": LogisticRegression(max_iter=1000),
                "logistic_regression": LogisticRegression(max_iter=1000),
                "decision_tree": DecisionTreeClassifier(random_state=42),
                "tree": DecisionTreeClassifier(random_state=42),
                "knn": KNeighborsClassifier(n_neighbors=5),
                "k_neighbors": KNeighborsClassifier(n_neighbors=5),
                "svm": SVC(kernel="rbf", probability=True),
                "support_vector": SVC(kernel="rbf", probability=True),
                "naive_bayes": GaussianNB(),
                "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
                "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
                "adaboost": AdaBoostClassifier(random_state=42, n_estimators=100),
                "extra_trees": ExtraTreesClassifier(random_state=42, n_estimators=100),
            }
        }

        if problem_type not in models:
            raise ValueError("Unsupported problem type")

        if model_type not in models[problem_type]:
            raise ValueError(f"Unsupported model type: {model_type}")

        return models[problem_type][model_type]
    
    # ==========================================================
    # PARAMETER GRID
    # ==========================================================

    def _get_param_grid(self, model_type: str, problem_type: str):

        grids = {

            "regression": {
                "linear": None,
                "ridge": {
                    "model__alpha": [0.1, 1.0, 10.0]
                },
                "lasso": {
                    "model__alpha": [0.001, 0.01, 0.1]
                },
                "decision_tree": {
                    "model__max_depth": [3, 5, 10, None],
                    "model__min_samples_split": [2, 5]
                },
                "knn": {
                    "model__n_neighbors": [3, 5, 7, 9],
                    "model__weights": ["uniform", "distance"]
                },
                "svr": {
                    "model__C": [0.1, 1, 10],
                    "model__gamma": ["scale", "auto"]
                },
                "random_forest": {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [5, 10, None],
                    "model__min_samples_split": [2, 5]
                },
                "gradient_boosting": {
                    "model__n_estimators": [50, 100, 150],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7]
                },
                "adaboost": {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.5, 1.0, 1.5]
                },
                "extra_trees": {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [5, 10, None]
                }
            },

            "classification": {
                "logistic": {
                    "model__C": [0.1, 1, 10],
                    "model__solver": ["lbfgs", "saga"]
                },
                "decision_tree": {
                    "model__max_depth": [3, 5, 10, None],
                    "model__min_samples_split": [2, 5]
                },
                "knn": {
                    "model__n_neighbors": [3, 5, 7, 9],
                    "model__weights": ["uniform", "distance"]
                },
                "svm": {
                    "model__C": [0.1, 1, 10],
                    "model__gamma": ["scale", "auto"]
                },
                "naive_bayes": None,
                "random_forest": {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [5, 10, None],
                    "model__min_samples_split": [2, 5]
                },
                "gradient_boosting": {
                    "model__n_estimators": [50, 100, 150],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7]
                },
                "adaboost": {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.5, 1.0, 1.5]
                },
                "extra_trees": {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [5, 10, None]
                }
            }
        }

        return grids.get(problem_type, {}).get(model_type, None)


    # ==========================================================
    # FEATURE TYPE DETECTION
    # ==========================================================

    def _detect_feature_types(self, X: pd.DataFrame):

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Attempt datetime detection
        datetime_features = []
        for col in X.columns:
            if col in numeric_features or col in categorical_features:
                continue
            try:
                parsed = pd.to_datetime(X[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    datetime_features.append(col)
            except:
                continue

        return numeric_features, categorical_features, datetime_features

    # ==========================================================
    # BUILD PREPROCESSOR
    # ==========================================================

    def _build_preprocessor(self, X: pd.DataFrame):

        numeric_features, categorical_features, datetime_features = \
            self._detect_feature_types(X)

        transformers = []

        if numeric_features:
            transformers.append((
                "num",
                StandardScaler(),
                numeric_features
            ))

        if categorical_features:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ))

        # Convert datetime columns to numeric timestamps
        for col in datetime_features:
            X[col] = pd.to_datetime(X[col], errors="coerce").astype("int64")

        if datetime_features:
            transformers.append((
                "dt",
                StandardScaler(),
                datetime_features
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return preprocessor, numeric_features, categorical_features, datetime_features

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

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column]

        if X.empty:
            raise ValueError("No features available for training.")

        preprocessor, num_feats, cat_feats, dt_feats = \
            self._build_preprocessor(X)

        model = self._get_model(model_type, problem_type)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        model_id = str(uuid.uuid4())

        ModelRegistry.register(
            model_id,
            pipeline,
                {
                    "problem_type": problem_type,
                    "dataset_id": dataset_id,
                    "target": target_column,
                    "model_type": model_type,
                    "numeric_features": num_feats,
                    "categorical_features": cat_feats,
                    "datetime_features": dt_feats,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test)
                }
        )

        return {
            "status": "success",
            "tool": "train_model",
            "result": {
                "model_id": model_id,
                "model_type": model_type,
                "problem_type": problem_type,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "numeric_features": num_feats,
                "categorical_features": cat_feats,
                "datetime_features": dt_feats
            }
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
    
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column]
    
        preprocessor, num_feats, cat_feats, dt_feats = \
            self._build_preprocessor(X)
    
        base_model = self._get_model(model_type, problem_type)
    
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", base_model)
        ])
    
        param_grid = self._get_param_grid(model_type, problem_type)
    
        if param_grid is None:
            raise ValueError(f"No tuning grid defined for model '{model_type}'")
    
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            n_jobs=-1
        )
    
        grid.fit(X, y)
    
        best_model = grid.best_estimator_
        model_id = str(uuid.uuid4())
    
        ModelRegistry.register(
            model_id,
            {
                "model": best_model,
                "metadata": {
                    "problem_type": problem_type,
                    "dataset_id": dataset_id,
                    "target": target_column,
                    "model_type": model_type,
                    "best_params": grid.best_params_,
                    "numeric_features": num_feats,
                    "categorical_features": cat_feats,
                    "datetime_features": dt_feats
                }
            }
        )
    
        return {
            "status": "success",
            "tool": "tune_model",
            "result": {
                "model_id": model_id,
                "best_params": grid.best_params_
            }
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

        pipeline = model_entry["model"]
        metadata = model_entry["metadata"]

        feature_columns = (
            metadata["numeric_features"] +
            metadata["categorical_features"] +
            metadata["datetime_features"]
        )

        row = {col: input_data.get(col) for col in feature_columns}
        X_input = pd.DataFrame([row])

        prediction = pipeline.predict(X_input)[0]

        if isinstance(prediction, (np.floating, np.integer)):
            prediction = float(prediction)

        return {
            "status": "success",
            "tool": "predict",
            "result": {
                "model_id": model_id,
                "prediction": prediction
            }
        }

    # ==========================================================
    # LIST AVAILABLE MODELS
    # ==========================================================

    def list_available_models(self) -> Dict[str, Any]:
        """Returns list of all available models for regression and classification"""
        return {
            "regression": {
                "traditional": ["linear", "ridge", "lasso"],
                "tree_based": ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "adaboost"],
                "instance_based": ["knn"],
                "kernel_based": ["svr"]
            },
            "classification": {
                "linear": ["logistic"],
                "tree_based": ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "adaboost"],
                "instance_based": ["knn", "naive_bayes"],
                "kernel_based": ["svm"]
            }
        }
