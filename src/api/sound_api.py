from fastapi import APIRouter, FastAPI, File, UploadFile
from api_dto import *
import os
import shutil
from service.voice_model import voice_model
from service.ai_model import main_model
from service.logger_setup import logger

router = APIRouter(
    prefix="/sound",
    tags=["sound"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def root():
    logger.info("Sound API root accessed.")
    return {"message": "Upload an MP3 file to /summarize to get transcript and summary."}

@router.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribes an uploaded audio file using the Whisper model.
    """

    if not audio_file.filename:
        return {
            "status_code": 400,
            "content": {"message": "No file uploaded."}
        }

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, audio_file.filename)

    # Get file size (Bytes)
    audio_file.file.seek(0, os.SEEK_END)   # move to end
    size_bytes = audio_file.file.tell()
    audio_file.file.seek(0)                # reset back to start

    # ext = audio_file.filename.split(".")[-1]
    # audio = AudioSegment.from_file(temp_file_path, format=ext)
    # duration_ms = len(audio)
    # minutes = duration_ms // 60000
    # seconds = (duration_ms % 60000) // 1000

    logger.info(f"Uploaded file: {audio_file.filename}")
    logger.info(f"Size: {size_bytes} bytes")
    # logger.info(f"Duration: {minutes}m {seconds}s")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        logger.info(f"Transcribing file: {audio_file.filename}")
        result = voice_model.transcribe(temp_file_path)

        logger.info(f"Transcription complete: {result}")
        return {
            "status_code": 200,
            "content": {"transcription": result}
        }
    except Exception as e:
        logger.info(f"An error occurred during transcription: {e}")
        return {
            "status_code": 500,
            "content": {"message": f"An error occurred: {e}"}
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

@router.post("/summarize")
async def summarize_audio(audio_file: UploadFile = File(...)):
    """
    Summary an audio with a few sentences
    """
    if not audio_file.filename:
        return {
            "status_code": 400,
            "content": {"message": "No file uploaded."}
        }

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, audio_file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        logger.info(f"Transcribing file: {audio_file.filename}")
        content = voice_model.transcribe(temp_file_path)

        logger.info(f"Transcription complete. Content: {content}")
        summarized_text = main_model.summarize(content)
        return {
            "status_code": 200,
            "content": {"summary": summarized_text}
        }
    except Exception as e:
        logger.info(f"An error occurred during transcription: {e}")
        return {
            "status_code": 500,
            "content": {"message": f"An error occurred: {e}"}
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)