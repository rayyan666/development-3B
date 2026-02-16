from fastapi import APIRouter, UploadFile, File
import shutil
import os
import pandas as pd

from app.api.schemas import ToolRequest
from app.core.orchestrator import Orchestrator
from app.state.dataset_registry import DatasetRegistry
from app.agent.chat_controller import ChatController
from pydantic import BaseModel

router = APIRouter()
orchestrator = Orchestrator()
chat_controller = ChatController()

DATA_DIR = "uploaded_datasets"
os.makedirs(DATA_DIR, exist_ok=True)


class SetDatasetRequest(BaseModel):
    dataset_id: str


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):

    contents = await file.read()

    df = pd.read_csv(pd.io.common.BytesIO(contents))

    dataset_id = file.filename.replace(".csv", "")
    DatasetRegistry.register(dataset_id, df)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


@router.post("/invoke")
def invoke_tool(request: ToolRequest):
    return orchestrator.handle(
        tool_name=request.tool_name,
        parameters=request.parameters
    )

@router.get("/datasets")
def list_datasets():
    return {
        "status": "success",
        "datasets": DatasetRegistry.list_datasets()
    }

@router.get("/preview/{dataset_id}")
def preview_dataset(dataset_id: str):

    if not DatasetRegistry.exists(dataset_id):
        return {"status": "error", "message": "Dataset not found"}

    df = DatasetRegistry.get(dataset_id)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(20).to_dict(orient="records")
    }

@router.get("/profile/{dataset_id}")
def profile_dataset(dataset_id: str):

    if not DatasetRegistry.exists(dataset_id):
        return {"status": "error", "message": "Dataset not found"}

    df = DatasetRegistry.get(dataset_id)

    from app.engines.data_profiler import DataProfiler

    profiler = DataProfiler(df)
    profile = profiler.profile()

    return {
        "status": "success",
        "profile": profile
    }

@router.post("/chat")
def chat_endpoint(payload: dict):
    message = payload.get("message")
    response = chat_controller.handle(message)
    return {"response": response}

@router.post("/set-active-dataset")
def set_active_dataset(request: SetDatasetRequest):

    dataset_id = request.dataset_id

    if not DatasetRegistry.exists(dataset_id):
        return {
            "status": "error",
            "message": f"Dataset '{dataset_id}' not found."
        }

    chat_controller.memory.last_dataset_id = dataset_id

    return {
        "status": "success",
        "active_dataset": dataset_id
    }

