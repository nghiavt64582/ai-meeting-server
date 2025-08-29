from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, FastAPI, File, UploadFile

class SummaryResponse(BaseModel):
    transcript: str
    summary: str

class AnswerResponse(BaseModel):
    answer: str

class SummaryRequest(BaseModel):
    text: str
    n_tokens: Optional[int] = 100

class SummaryAudioReq(BaseModel):
    audio_file: UploadFile = File(...)
    summary_prompt: Optional[str] = "Summarize this conversation in a few sentences :"