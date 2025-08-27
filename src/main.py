from fastapi import FastAPI
from api import ai_model_api
from api import admin_api
from api import sound_api

app = FastAPI(
    title="My Modular FastAPI App",
    description="A demonstration of structuring FastAPI routes into multiple files.",
    version="0.1.0",
)

app.include_router(ai_model_api.router)
app.include_router(admin_api.router)
app.include_router(sound_api.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the root of the modular FastAPI app!"}