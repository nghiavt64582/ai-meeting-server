from fastapi import APIRouter, File, UploadFile
import time
from service.voice_model import voice_model
from service.ai_model import ai_model
from service.logger_setup import logger
from api_dto import *

router = APIRouter(
    prefix="/sound",
    tags=["sound"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def root():
    logger.info("Sound API root accessed.")
    return {"message": "Upload an MP3 file to /summarize to get transcript and summary."}

@router.post("/transcribe-audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...)
):
    """
    Transcribes an uploaded audio file using the Whisper model.
    """
    start_time = time.time()
    transcribed_data = voice_model.transcribe_audio(audio_file)
    content = transcribed_data.get("content", "")
    number_of_words = content.count(' ') + content.count('.')
    logger.info(f"Transcription took {time.time() - start_time:.2f} seconds. Words: {number_of_words}, WPS: {number_of_words / (time.time() - start_time):.2f}")
    if content:
        return {
            "status_code": 200,
            "content": {
                "transcription": content,
                "number_of_words": number_of_words
            }
        }
    else:
        return {
            "status_code": 500,
            "content": {"message": "An error occurred during transcription."}
        }

@router.post("/summarize-audio")
async def summarize_audio(
    audio_file: UploadFile = File(...), 
    summary_prompt: Optional[str] = "Summarize this conversation in a few sentences :",
    model: str = "os"
):
    """
    Summary an audio with a few sentences
    """

    start_time = time.time()
    transcribed_data = voice_model.transcribe_audio(audio_file)
    content = transcribed_data.get("content", "")
    number_of_words = content.count(' ') + content.count('.')
    time_1 = time.time()
    logger.info(f"Transcription took {time_1 - start_time:.2f} seconds. Words: {number_of_words}, WPS: {number_of_words / (time_1 - start_time):.2f}")
    logger.info(f"Starting summarization with prompt: {summary_prompt}, content {content}")
    summarized_text = ai_model.summarize(text=content, summary_prompt=summary_prompt, is_use_gemini=model=="gemini")
    logger.info(f"Summarization took {time.time() - time_1:.2f} seconds. Words: {number_of_words}, WPS: {number_of_words / (time.time() - time_1):.2f}")

    if content:
        return {
            "status_code": 200,
            "content": {
                "transcription": content,
                "summary": summarized_text,
                "number_of_words": number_of_words
            }
        }
    else:
        return {
            "status_code": 500,
            "content": {"message": "An error occurred during transcription."}
        }
    
@router.post("/summarize-text")
async def summarize_text(
    content: str, 
    summary_prompt: str = "Summarize this conversation in a few sentences :",
    model: str = "os"
):
    start_time = time.time()
    summarized_text = ai_model.summarize(text=content, summary_prompt=summary_prompt, is_use_gemini=model=="gemini")
    number_of_words = content.count(' ') + content.count('.')
    logger.info(f"Summarization took {time.time() - start_time:.2f} seconds. Words: {number_of_words}, WPS: {number_of_words / (time.time() - start_time):.2f}")

    if summarized_text:
        return {
            "status_code": 200,
            "content": {
                "summary": summarized_text,
                "number_of_words": number_of_words
            }
        }
    else:
        return {
            "status_code": 500,
            "content": {"message": "An error occurred during summarization."}
        }