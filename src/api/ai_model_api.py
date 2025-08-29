from fastapi import APIRouter
from api_dto import *
from service.ai_model import ai_model


router = APIRouter(
    prefix="/ai-model",
    tags=["ai-model"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def root():
    return {"message": "Send a request to the AI model."}

@router.post("/summarize-text", response_model=SummaryResponse)
async def summarize_text(req: SummaryRequest):
    text = req.text.strip()
    answer = ai_model.generate_response(f"Summary this conversation : {text}")
    return SummaryResponse(transcript=text, summary=answer)

@router.post("/question", response_model=AnswerResponse)
async def answer_questions(req: SummaryRequest):
    answer = ai_model.generate_response(req.text, req.n_tokens)
    return AnswerResponse(answer=answer)