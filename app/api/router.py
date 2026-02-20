from fastapi import APIRouter, UploadFile, File
import os
import pandas as pd

from app.api.schemas import ToolRequest
from app.core.orchestrator import Orchestrator
from app.state.dataset_registry import DatasetRegistry
from app.state.model_registry import ModelRegistry
from app.core.session import chat_controller
from pydantic import BaseModel

router = APIRouter()
orchestrator = Orchestrator()

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

    result = orchestrator.handle(
        tool_name=request.tool_name,
        parameters=request.parameters
    )

    # ---- Keep memory in sync ----
    if request.tool_name == "train_model":
        try:
            model_id = result["result"]["result"]["model_id"]
            chat_controller.memory.last_model_id = model_id
        except:
            pass

    if request.tool_name == "load_csv":
        try:
            dataset_id = result["result"]["result"]["dataset_id"]
            chat_controller.memory.last_dataset_id = dataset_id
        except:
            pass

    return result



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

    from app.engines.data_profiler import DataProfiler

    df = DatasetRegistry.get(dataset_id)
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


@router.get("/inspector")
def get_inspector_state():

    dataset_id = chat_controller.memory.last_dataset_id
    model_id = chat_controller.memory.last_model_id

    dataset_info = None
    model_info = None

    if dataset_id and DatasetRegistry.exists(dataset_id):
        df = DatasetRegistry.get(dataset_id)
        dataset_info = {
            "dataset_id": dataset_id,
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    if model_id and ModelRegistry.exists(model_id):
        model_entry = ModelRegistry.get(model_id)
        model_info = {
            "model_id": model_id,
            "problem_type": model_entry["metadata"]["problem_type"],
            "target": model_entry["metadata"]["target"],
            "model_type": model_entry["metadata"]["model_type"]
        }

    return {
        "dataset": dataset_info,
        "model": model_info
    }


@router.get("/models/available")
def get_available_models():
    from app.engines.ml_engine import MLEngine
    
    engine = MLEngine()
    available_models = engine.list_available_models()
    
    return {
        "status": "success",
        "models": available_models
    }


@router.get("/hyperparameters/{problem_type}/{model_type}")
def get_hyperparameters(problem_type: str, model_type: str):
    """Get hyperparameter ranges for a specific model"""
    from app.engines.ml_engine import MLEngine
    
    engine = MLEngine()
    param_grid = engine._get_param_grid(model_type, problem_type)
    
    return {
        "status": "success",
        "model_type": model_type,
        "problem_type": problem_type,
        "hyperparameters": param_grid or {}
    }
