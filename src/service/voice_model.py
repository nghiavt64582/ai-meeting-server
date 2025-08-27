import whisper
import logging

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

voice_model = VoiceModel(model_id="base", device="cpu")