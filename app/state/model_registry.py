from threading import Lock
from typing import Dict, Any, List


class ModelNotFound(Exception):
    pass


class ModelRegistry:
    """
    In-memory model registry.
    Stores trained model objects with metadata.
    """

    _models: Dict[str, Dict[str, Any]] = {}
    _lock = Lock()

    @classmethod
    def register(cls, model_id: str, model_obj: Any, metadata: Dict[str, Any]) -> None:
        with cls._lock:
            cls._models[model_id] = {
                "model": model_obj,
                "metadata": metadata
            }

    @classmethod
    def get(cls, model_id: str) -> Dict[str, Any]:
        model_entry = cls._models.get(model_id)
        if model_entry is None:
            raise ModelNotFound(f"Model '{model_id}' not found.")
        return model_entry

    @classmethod
    def exists(cls, model_id: str) -> bool:
        return model_id in cls._models

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())

    @classmethod
    def remove(cls, model_id: str) -> None:
        with cls._lock:
            if model_id in cls._models:
                del cls._models[model_id]
