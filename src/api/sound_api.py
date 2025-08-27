from fastapi import APIRouter, FastAPI, File, UploadFile
from api_dto import *
import os
import shutil
from voice_model import voice_model
from ai_model import main_model
import traceback

router = APIRouter(
    prefix="/sound",
    tags=["sound"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def root():
    return {"message": "Upload an MP3 file to /summarize to get transcript and summary."}

@router.get("/")
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the Whisper Audio Transcriber API. Use the /transcribe endpoint to upload an audio file."}

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

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        print(f"Transcribing file: {audio_file.filename}")
        result = voice_model.transcribe(temp_file_path)

        print(f"Transcription complete: {result}")
        return {
            "status_code": 200,
            "content": {"transcription": result}
        }
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
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

        print(f"Transcribing file: {audio_file.filename}")
        content = voice_model.transcribe(temp_file_path)

        print(f"Transcription complete. Content: {content}")
        summarized_text = main_model.summarize(content)
        return {
            "status_code": 200,
            "content": {"summary": summarized_text}
        }
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return {
            "status_code": 500,
            "content": {"message": f"An error occurred: {e}"}
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)