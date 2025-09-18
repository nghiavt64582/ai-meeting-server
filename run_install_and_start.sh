#!/bin/bash
set -e  # dừng script nếu có lỗi

# --- Update hệ thống ---
sudo apt update -y

# --- Cài các công cụ cần thiết ---
sudo apt install -y software-properties-common

# --- Thêm repo Python 3.12 ---
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y

# --- Cài Python 3.12 và công cụ ---
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# --- Tạo virtual environment ---
python3.12 -m venv .venv
source .venv/bin/activate

# --- Cài ffmpeg ---
sudo apt install -y ffmpeg

# --- Cài PyTorch GPU ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Cài các thư viện Python khác ---
pip install fastapi huggingface_hub openai_whisper pydantic requests transformers whisperx whisper uvicorn dotenv accelerate openai-whisper python-multipart einops transformers_stream_generator

# --- Chạy FastAPI app ---
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ../.env --reload
