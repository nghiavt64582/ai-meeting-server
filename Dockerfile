# Base image
FROM python:3.12-slim

# Đặt biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài gói hệ thống cần thiết (ffmpeg, build tools, ...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước (tận dụng cache)
COPY requirements.txt .

# Cài các thư viện Python (bao gồm torch từ index PyTorch)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# Fix: ensure whisperx installed from GitHub
RUN pip install git+https://github.com/m-bain/whisperx.git@v3.4.2

RUN pip install accelerate transformers fastapi uvicorn openai-whisper python-multipart einops transformers_stream_generator whisperx dotenv

# Copy toàn bộ code vào container
COPY . .

# Chuyển vào src để chạy app
WORKDIR /app/src

# Expose port
EXPOSE 8000

# Chạy uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--env-file", "../.env"]