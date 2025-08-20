from pydantic import BaseModel
from typing import Optional

class SummaryResponse(BaseModel):
    transcript: str
    summary: str

class AnswerResponse(BaseModel):
    answer: str

class SummaryRequest(BaseModel):
    text: str
    n_tokens: Optional[int] = 100