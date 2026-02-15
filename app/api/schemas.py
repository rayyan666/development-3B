from pydantic import BaseModel
from typing import Dict, Any


class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolResponse(BaseModel):
    status: str
    tool: str
    result: Dict[str, Any] | None = None
    error: str | None = None
