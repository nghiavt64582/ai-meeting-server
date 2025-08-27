import whisper
import logging
from fastapi import APIRouter, FastAPI, File, UploadFile
import os
from service.logger_setup import logger
import shutil

class VoiceModel:
    """
    A class to handle voice model operations, including loading the model and generating responses.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        try:
            self.whisper_model = whisper.load_model("base")
        except Exception as e:
            logging.error("Failed to load Whisper model: %s", e)
            raise
    
    def transcribe(self, audio_path: str) -> str:
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result['text']
        except Exception as e:
            logging.error("Error during transcription: %s", e)
            return ""

    def transcribe_audio(self, audio_file: UploadFile = File(...)):
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
            result = self.transcribe(temp_file_path)

            return result
        except Exception as e:
            logger.info(f"An error occurred during transcription: {e}")
            return ""
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)


voice_model = VoiceModel(model_id="base", device="cpu")