import whisper
import logging
from fastapi import APIRouter, FastAPI, File, UploadFile
import os
from service.logger_setup import logger
import shutil
import subprocess
import json

class VoiceModel:
    """
    A class to handle voice model operations, including loading the model and generating responses.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        # os.environ["PATH"] += r";C:\Users\Administrator\Desktop\ffmpeg-7.1.1-essentials_build\bin"
        try:
            self.whisper_model = whisper.load_model("base", device=device)
        except Exception as e:
            logging.error("Failed to load Whisper model: %s", e)
            raise
    
    def transcribe_audio(self, audio_file: UploadFile = File(...)):
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, audio_file.filename)


        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            size_st = self.get_file_size(audio_file)
            logger.info(f"Uploaded file: {audio_file.filename}, size {size_st}")

            logger.info(f"Transcribing file: {audio_file.filename}")
            content = self.whisper_model.transcribe(temp_file_path, language="en")['text']

            duration = self.get_duration(temp_file_path)
            logger.info(f"Duration of file: {audio_file.filename}, duration {duration} s")
            return {
                "content": content,
                "size": size_st,
                "duration": duration
            }
        except Exception as e:
            logger.info(f"An error occurred during transcription: {e}")
            return ""
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)

    def get_file_size(self, file: UploadFile) -> str:
        # return file size with nearest units (B, KB, MB)
        file.file.seek(0, os.SEEK_END)
        size_bytes = file.file.tell()
        file.file.seek(0)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / 1024**2:.2f} MB"
        
    def get_duration(self, filepath: str) -> int:
        """
        Gets the duration of a media file (audio or video) using ffprobe.
        Requires FFmpeg (which includes ffprobe) to be installed on the system.
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return None

        try:
            subprocess.run(['ffprobe', '-h'], capture_output=True, check=True, text=True)
        except FileNotFoundError:
            print("Error: 'ffprobe' command not found. Please install FFmpeg and ensure it's in your system's PATH.")
            print("Installation instructions for FFmpeg can be found below.")
            return None
        except subprocess.CalledProcessError as e:
            if "ffprobe version" not in e.stdout:
                print(f"Warning: ffprobe check returned an unexpected error: {e.stderr}")
                pass

        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                filepath 
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
            
            data = json.loads(result.stdout)
            
            duration_str = data.get('format', {}).get('duration')
            if duration_str:
                return int(float(duration_str))
            else:
                print(f"Error: Could not find 'duration' in ffprobe output for {filepath}. Output: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Error running ffprobe for '{filepath}': {e}")
            print(f"ffprobe stderr: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing ffprobe JSON output for '{filepath}': {e}. Output: {result.stdout}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filepath}': {e}")
            return None


voice_model = VoiceModel(model_id="base", device="cpu")