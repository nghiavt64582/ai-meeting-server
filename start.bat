REM Cài đặt FFmpeg (thủ công)
echo.
echo =========================================================
echo Cài đặt FFmpeg:
echo Vui lòng tải và cài đặt FFmpeg từ trang web chính thức.
echo Download link: https://ffmpeg.org/download.html
echo Sau khi tải về, hãy thêm thư mục 'bin' của FFmpeg vào biến môi trường PATH.
echo =========================================================
echo.

@echo off
REM Thiết lập cửa sổ Command Prompt
title Setup and Run Python App

REM Tạo và kích hoạt môi trường ảo
echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .\.venv\Scripts\activate

REM Cài đặt các thư viện cần thiết
echo Installing required libraries from requirements.txt...
pip install -r requirements.txt

echo Installing additional libraries...
pip install accelerate transformers fastapi uvicorn openai-whisper python-multipart einops transformers_stream_generator hf_xet optimum

echo Installing torch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM vào src và Chạy ứng dụng FastAPI (trạng thái không chờ giống nohup trên Linux)
echo.
echo Starting FastAPI application...
cd src
start /B uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Giữ cửa sổ mở sau khi chạy
pause
