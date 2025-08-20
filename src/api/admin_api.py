from fastapi import APIRouter
from api_dto import *
from voice_model import *
from ai_model import *
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
    main_model.load_model(model_id)
    return {"message": f"Model set to {model_id}"}