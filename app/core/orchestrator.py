from typing import Dict, Any
from app.core.dispatcher import Dispatcher


class Orchestrator:

    def __init__(self):
        self.dispatcher = Dispatcher()

    def handle(self, tool_name: str, parameters: Dict[str, Any]):

        try:
            result = self.dispatcher.dispatch(tool_name, parameters)

            return {
                "status": "success",
                "tool": tool_name,
                "result": result
            }

        except Exception as e:
            return {
                "status": "error",
                "tool": tool_name,
                "error": str(e)
            }
