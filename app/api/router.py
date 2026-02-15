from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.api.schemas import ToolRequest
from app.core.orchestrator import Orchestrator

router = APIRouter()
orchestrator = Orchestrator()

DATA_DIR = "uploaded_datasets"
os.makedirs(DATA_DIR, exist_ok=True)


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "uploaded", "path": file_path}


@router.post("/invoke")
def invoke_tool(request: ToolRequest):
    return orchestrator.handle(
        tool_name=request.tool_name,
        parameters=request.parameters
    )
