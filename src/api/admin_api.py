from fastapi import APIRouter
from service.ai_model import *
from service.voice_model import *
from popular_models import popular_ai_models

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def root():
    return {"message": "Admin panel for managing AI models."}

@router.get("/models")
async def list_models():
    return {"models": [model["id"] for model in popular_ai_models]}

@router.post("/model")
async def set_model(model_id: str):
    # check if model_id is valid
    if model_id not in [model["id"] for model in popular_ai_models]:
        return {"error": "Invalid model ID"}
    # set model
    ai_model.load_model(model_id)
    return {"message": f"Model set to {model_id}"}

@router.post("/preload-model")
async def preload_model(model_id: str):
    if model_id not in [model["id"] for model in popular_ai_models]:
        return {"error": "Invalid model ID"}

    ai_model.preload_model(model_id)
    return {"message": f"Model {model_id} preloaded successfully."}