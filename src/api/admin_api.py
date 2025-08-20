from fastapi import APIRouter
from api_dto import *
from voice_model import *
from ai_model import *

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
    return {"models": popular_ai_models}

@router.post("/model")
async def set_model(model_id: str):
    return {"message": f"Model set to {model_id}"}