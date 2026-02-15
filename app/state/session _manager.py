from threading import Lock
from typing import Optional, Dict


class SessionManager:
    """
    Simple in-memory session tracking.
    Stores lightweight conversation state.
    """

    _sessions: Dict[str, Dict[str, str]] = {}
    _lock = Lock()

    @classmethod
    def initialize_session(cls, session_id: str) -> None:
        with cls._lock:
            if session_id not in cls._sessions:
                cls._sessions[session_id] = {
                    "last_dataset": None,
                    "last_model": None
                }

    @classmethod
    def set_last_dataset(cls, session_id: str, dataset_id: str) -> None:
        with cls._lock:
            cls.initialize_session(session_id)
            cls._sessions[session_id]["last_dataset"] = dataset_id

    @classmethod
    def get_last_dataset(cls, session_id: str) -> Optional[str]:
        cls.initialize_session(session_id)
        return cls._sessions[session_id]["last_dataset"]

    @classmethod
    def set_last_model(cls, session_id: str, model_id: str) -> None:
        with cls._lock:
            cls.initialize_session(session_id)
            cls._sessions[session_id]["last_model"] = model_id

    @classmethod
    def get_last_model(cls, session_id: str) -> Optional[str]:
        cls.initialize_session(session_id)
        return cls._sessions[session_id]["last_model"]

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        with cls._lock:
            if session_id in cls._sessions:
                del cls._sessions[session_id]
