import pandas as pd
import numpy as np
import warnings


class DataProfiler:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # UTILITY: SANITIZE FLOATS FOR JSON SERIALIZATION
    # --------------------------------------------------

    def _sanitize_float(self, value):
        """Convert NaN/inf to None for JSON serialization"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------

    def profile(self):
        profile = {
            "dataset_summary": self._dataset_summary(),
            "column_profile": self._column_profiles(),
            "numeric_columns": self._numeric_columns(),
            "categorical_columns": self._categorical_columns(),
            "high_cardinality_columns": self._high_cardinality(),
            "time_columns": self._time_columns(),
            "target_candidates": self._target_candidates(),
            "multicollinearity": self._multicollinearity(),
            "correlation_matrix": self._correlation_matrix(),
            "warnings": self._warnings()
        }

        return profile

    # --------------------------------------------------
    # BASIC SUMMARY
    # --------------------------------------------------

    def _dataset_summary(self):
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1])
        }

    # --------------------------------------------------
    # COLUMN PROFILES
    # --------------------------------------------------

    def _column_profiles(self):

        profiles = {}

        for col in self.df.columns:

            series = self.df[col]
            missing_pct = float(series.isna().mean())

            profile = {
                "dtype": str(series.dtype),
                "missing_pct": round(missing_pct, 4),
                "unique_values": int(series.nunique())
            }

            if pd.api.types.is_numeric_dtype(series):
                profile.update(self._numeric_stats(series))

            profiles[col] = profile

        return profiles

    # --------------------------------------------------
    # NUMERIC STATS
    # --------------------------------------------------

    def _numeric_stats(self, series):

        if series.dropna().empty:
            return {}

        return {
            "mean": self._sanitize_float(series.mean()),
            "std": self._sanitize_float(series.std()),
            "min": self._sanitize_float(series.min()),
            "max": self._sanitize_float(series.max()),
            "skewness": self._sanitize_float(series.skew())
        }

    # --------------------------------------------------
    # NUMERIC / CATEGORICAL DETECTION
    # --------------------------------------------------

    def _numeric_columns(self):
        return list(self.df.select_dtypes(include=["number"]).columns)

    def _categorical_columns(self):
        return list(self.df.select_dtypes(include=["object", "category"]).columns)

    # --------------------------------------------------
    # HIGH CARDINALITY DETECTION
    # --------------------------------------------------

    def _high_cardinality(self):

        high_card_cols = []

        for col in self._categorical_columns():
            unique_ratio = self.df[col].nunique() / len(self.df)

            if unique_ratio > 0.5:
                high_card_cols.append(col)

        return high_card_cols

    # --------------------------------------------------
    # TIME COLUMN DETECTION
    # --------------------------------------------------

    def _time_columns(self):

        time_cols = []

        for col in self.df.columns:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(self.df[col], errors="coerce")

                if parsed.notna().mean() > 0.8:
                    time_cols.append(col)
            except Exception:
                continue

        return time_cols

    # --------------------------------------------------
    # TARGET CANDIDATE INFERENCE
    # --------------------------------------------------

    def _target_candidates(self):

        numeric_cols = self._numeric_columns()
        categorical_cols = self._categorical_columns()

        regression_candidates = []
        classification_candidates = []

        for col in numeric_cols:

            unique_ratio = self.df[col].nunique() / len(self.df)

            # Continuous numeric
            if unique_ratio > 0.1:
                regression_candidates.append(col)

            # Low-unique numeric → possible classification
            if self.df[col].nunique() <= 10:
                classification_candidates.append(col)

        for col in categorical_cols:

            unique_count = self.df[col].nunique()

            if unique_count <= 10:
                classification_candidates.append(col)

        return {
            "regression": regression_candidates,
            "classification": classification_candidates
        }

    # --------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------

    def _correlation_matrix(self):

        numeric_df = self.df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()
        
        # Sanitize correlation values to handle NaN
        sanitized = {}
        for col in corr_matrix.columns:
            sanitized[col] = {}
            for row in corr_matrix.index:
                value = corr_matrix.loc[row, col]
                sanitized[col][row] = self._sanitize_float(value)
        
        return sanitized

    # --------------------------------------------------
    # MULTICOLLINEARITY DETECTION
    # --------------------------------------------------

    def _multicollinearity(self):

        numeric_df = self.df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr().abs()

        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                corr_value = corr_matrix.iloc[i, j]
                if pd.notna(corr_value) and corr_value > 0.9:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": self._sanitize_float(corr_value)
                    })

        return high_corr_pairs

    # --------------------------------------------------
    # WARNINGS
    # --------------------------------------------------

    def _warnings(self):

        warnings = []

        # Small dataset warning
        if len(self.df) < 100:
            warnings.append("Dataset is small. Risk of overfitting.")

        # High missing values warning
        for col in self.df.columns:
            if self.df[col].isna().mean() > 0.4:
                warnings.append(f"Column '{col}' has high missing values.")

        # High cardinality warning
        for col in self._high_cardinality():
            warnings.append(f"Column '{col}' has high cardinality.")

        return warnings
