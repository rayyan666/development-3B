from threading import Lock
from typing import Dict, List
import pandas as pd


class DatasetNotFound(Exception):
    pass


class DatasetRegistry:
    """
    Thread-safe in-memory dataset registry.
    Stores pandas DataFrames directly by dataset_id.
    """

    _datasets: Dict[str, pd.DataFrame] = {}
    _lock = Lock()

    # --------------------------------------------------
    # Register Dataset
    # --------------------------------------------------

    @classmethod
    def register(cls, dataset_id: str, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Only pandas DataFrame objects can be registered.")

        with cls._lock:
            cls._datasets[dataset_id] = dataframe

    # --------------------------------------------------
    # Get Dataset
    # --------------------------------------------------

    @classmethod
    def get(cls, dataset_id: str) -> pd.DataFrame:
        dataset = cls._datasets.get(dataset_id)

        if dataset is None:
            raise DatasetNotFound(f"Dataset '{dataset_id}' not found.")

        return dataset

    # --------------------------------------------------
    # Check Existence
    # --------------------------------------------------

    @classmethod
    def exists(cls, dataset_id: str) -> bool:
        return dataset_id in cls._datasets

    # --------------------------------------------------
    # List All Datasets
    # --------------------------------------------------

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._datasets.keys())

    # --------------------------------------------------
    # Remove Dataset
    # --------------------------------------------------

    @classmethod
    def remove(cls, dataset_id: str) -> None:
        with cls._lock:
            if dataset_id not in cls._datasets:
                raise DatasetNotFound(f"Dataset '{dataset_id}' not found.")

            del cls._datasets[dataset_id]

    # --------------------------------------------------
    # Clear All Datasets (Optional Utility)
    # --------------------------------------------------

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._datasets.clear()
