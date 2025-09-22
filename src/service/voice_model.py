import logging
from fastapi import APIRouter, FastAPI, File, UploadFile
import os
from service.logger_setup import logger
import shutil
import subprocess
import json
import requests
import base64
import mimetypes
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
from functools import lru_cache
class VoiceModel:
    """
    A class to handle voice model operations, including loading the model and generating responses.
    """
    def __init__(self):
        # os.environ["PATH"] += r";C:\Users\Administrator\Desktop\ffmpeg-7.1.1-essentials_build\bin"
        self.gemini_api_key = os.getenv("gemini_api_key", "").strip()
        logger.info(f"Gemini API Key: {self.gemini_api_key}")

        self.device, self.compute = self.pick_device_and_compute()
        logger.info(f"Using device: {self.device}, compute: {self.compute}")
        try:
            self.whisper_model = whisperx.load_model(
                "base", 
                compute_type=self.compute,
                device=self.device
            )
        except Exception as e:
            logging.error("Failed to load Whisper model: %s", e)
            raise

        # set up diarization model
        # load by key hf_token, the file .env is in previous of root folder
        self.hf_token = os.getenv("hf_token", "").strip()
        if not self.hf_token:
            raise RuntimeError("Set biến môi trường HF_TOKEN trước khi chạy diarization.")
        else:
            logger.info(f"HF_TOKEN found, proceeding with diarization. Hf token: {self.hf_token}   ")
        self.diarize_model = DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.device  # "cpu"
        )

    def calculate_blocks(self, segments, max_block_size=500, max_chars=300, gap_threshold=1.0):
        blocks = []
        cur = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"].strip()
        }

        for s in segments[1:]:
            gap = s["start"] - cur["end"]
            if gap > gap_threshold or len(cur["text"]) >= max_chars:
                blocks.append(cur)
                cur = {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip()
                }
            else:
                cur["end"] = s["end"]
                cur["text"] += (" " if cur["text"] else "") + s["text"].strip()

        blocks.append(cur)
        return blocks
    
    def transcribe_audio(self, audio_file: UploadFile = File(...), is_use_gemini: bool = False):
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, audio_file.filename)

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            size_st = self.get_file_size(audio_file)
            logger.info(f"Uploaded file: {audio_file.filename}, size {size_st}")

            logger.info(f"Transcribing file: {audio_file.filename}")
            if is_use_gemini:
                content = self.transcribe_audio_by_gemini_api(temp_file_path)
            else:
                audio = whisperx.load_audio(temp_file_path)
                cur_result = self.whisper_model.transcribe(
                    audio,
                    language="en",
                    batch_size=16
                )
                align_model, metadata = whisperx.load_align_model(
                    language_code=cur_result["language"],
                    device=self.device
                )
                result_aligned = whisperx.align(
                    cur_result["segments"], 
                    align_model, 
                    metadata, 
                    audio,
                    device=self.device,
                    return_char_alignments=False
                )
                
                diarize_df = self.diarize_model(
                    temp_file_path,
                    min_speakers=2,
                    max_speakers=3
                )

                # 4) Gán speaker cho từng từ/segment và gộp thành lượt thoại
                with_speaker = whisperx.assign_word_speakers(diarize_df, result_aligned)
                content = " ".join(s.get("text", "").strip() for s in cur_result.get("segments", []) if s.get("text"))
                blocks = self.calculate_block_by_whisperx(cur_result["segments"], with_speaker)
                for block in blocks:
                    logger.info(f"{block['speaker']} : {block['start']:.1f} --> {block['end']:.1f} : {block['text']}")

            duration = self.get_duration(temp_file_path)
            logger.info(f"Duration : {duration} s")
            result = {
                "content": content,
                "size": size_st,
                "duration": duration,
                "blocks": blocks if not is_use_gemini else []
            }
            return result
        except Exception as e:
            logger.info(f"An error occurred during transcription: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
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

    def transcribe_audio_by_gemini_api(self, audio_file_path: str) -> str:
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        mime = mimetypes.guess_type(audio_file_path)[0] or "audio/wav"
        with open(audio_file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Transcribe the following audio verbatim. "
                                "Return only the transcript with no extra commentary."
                            )
                        },
                        {   # <-- audio goes here
                            "inline_data": {"mime_type": mime, "data": b64}
                        }
                    ],
                }
            ],
            # Optional but nice:
            "generationConfig": {"response_mime_type": "text/plain"}
        }

        r = requests.post(
            gemini_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.gemini_api_key,     # header is fine; query ?key=... also works
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()

        # extract first candidate text
        return data["candidates"][0]["content"]["parts"][0].get("text", "")
    
    def calculate_block_by_whisperx(
        self,
        segments,                 # không dùng nhiều, giữ để tương thích
        with_speaker: dict,
        max_block_size: int = 500,
        max_chars: int = 300,
        gap_threshold: float = 1.0
    ):
        """
        Gộp transcript thành các block theo người nói (speaker) dựa trên output đã gán speaker của whisperx.

        Params:
        - segments: list segment gốc của Whisper (để tương thích; có thể không dùng)
        - with_speaker: dict từ whisperx.assign_word_speakers(...), kỳ vọng có key "segments"
        - max_block_size: tối đa số từ trong 1 block
        - max_chars: tối đa số ký tự text trong 1 block
        - gap_threshold: nếu khoảng cách giữa 2 segment > ngưỡng (giây) thì tách block

        Return:
        - List[dict]: mỗi dict gồm {speaker, start, end, text, words, n_words}
        """
        diar_segments = (with_speaker or {}).get("segments", []) or []
        if not diar_segments:
            # fallback: không có diarization → gom thành 1 block, speaker=UNK
            joined = " ".join(s.get("text", "").strip() for s in (segments or []) if s.get("text"))
            if not joined:
                return []
            start = float((segments or [{}])[0].get("start", 0.0))
            end = float((segments or [{}])[-1].get("end", 0.0))
            return [{
                "speaker": "SPEAKER_UNK",
                "start": float(start),
                "end": float(end),
                "text": joined.strip(),
                "words": [],
                "n_words": 0,
            }]

        def _to_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        def _majority_speaker(words):
            # chọn speaker xuất hiện nhiều nhất, tie-break theo tổng duration
            if not words:
                return None
            from collections import Counter, defaultdict
            c = Counter()
            dur = defaultdict(float)
            for w in words:
                spk = w.get("speaker")
                if spk is None:
                    continue
                c[spk] += 1
                dur[spk] += max(0.0, _to_float(w.get("end")) - _to_float(w.get("start")))
            if not c:
                return None
            # ưu tiên theo count, rồi duration
            best = sorted(c.items(), key=lambda kv: (kv[1], dur.get(kv[0], 0.0)), reverse=True)[0][0]
            return best

        blocks = []
        cur = None

        for seg in diar_segments:
            if not seg:
                continue
            s_text = (seg.get("text") or "").strip()
            if not s_text:
                continue

            s_words = seg.get("words") or []
            s_speaker = seg.get("speaker") or _majority_speaker(s_words) or "SPEAKER_UNK"
            s_start = _to_float(seg.get("start"), 0.0)
            s_end = _to_float(seg.get("end"), s_start)

            # Khởi tạo block đầu tiên
            if cur is None:
                cur = {
                    "speaker": s_speaker,
                    "start": s_start,
                    "end": s_end,
                    "text": s_text,
                    "words": list(s_words),
                }
                continue

            gap = max(0.0, s_start - _to_float(cur["end"], s_start))

            # Điều kiện tách block
            need_split = (
                s_speaker != cur["speaker"] or
                gap > gap_threshold or
                (len(cur["text"]) + 1 + len(s_text) > max_chars) or
                (len(cur["words"]) + len(s_words) > max_block_size)
            )

            if need_split:
                # chốt block cũ
                cur["n_words"] = len(cur["words"])
                cur["start"] = _to_float(cur["start"])
                cur["end"] = _to_float(cur["end"])
                blocks.append(cur)

                # mở block mới
                cur = {
                    "speaker": s_speaker,
                    "start": s_start,
                    "end": s_end,
                    "text": s_text,
                    "words": list(s_words),
                }
            else:
                # gộp vào block hiện tại
                cur["end"] = max(_to_float(cur["end"]), s_end)
                # tránh 2 dấu cách
                cur["text"] = (cur["text"] + " " + s_text).strip()
                if s_words:
                    cur["words"].extend(s_words)

        # đẩy block cuối
        if cur is not None:
            cur["n_words"] = len(cur["words"])
            cur["start"] = _to_float(cur["start"])
            cur["end"] = _to_float(cur["end"])
            blocks.append(cur)

        # Chuẩn hoá: ép float cho start/end trong words (nếu bạn muốn JSON sạch)
        for b in blocks:
            for w in b["words"]:
                if "start" in w: w["start"] = _to_float(w["start"])
                if "end" in w:   w["end"]   = _to_float(w["end"])
        # tạm thời xóa key "words" để giảm dung lượng trả về
        for b in blocks:
            if "words" in b:
                del b["words"]
        return blocks

    def pick_device_and_compute(self):
        # Ưu tiên CUDA nếu CTranslate2 nhìn thấy GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            # Kiểm tra compute capability để quyết định FP16
            try:
                major, minor = torch.cuda.get_device_capability(0)
            except Exception:
                major, minor = (0, 0)

            # Turing/Ampere/Ada (>= 7.0) dùng FP16 hiệu quả
            precision = "float16" if major >= 7 else "float32"
        else:
            precision = "float32"

        return device, precision
    

@lru_cache(maxsize=1)
def get_voice_model():
    return VoiceModel()