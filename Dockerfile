# Base image: CUDA 12.1 + cuDNN8 runtime trên Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# --- Cài Python và gói hệ thống ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# --- Cài PyTorch GPU ---
# Lưu ý: cu121 = CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# --- Cài các thư viện Python khác ---
RUN pip install --no-cache-dir \
    fastapi \
    huggingface_hub \
    openai_whisper \
    pydantic \
    requests \
    transformers \
    whisperx \
    whisper \
    uvicorn \
    python-dotenv \
    accelerate \
    python-multipart \
    einops \
    transformers_stream_generator

# --- Fix executable stack cho libctranslate2 bằng patchelf ---
RUN find /usr/local/lib/python3.12/site-packages -name "libctranslate2*.so*" \
    -exec patchelf --clear-execstack {} \; || true

# Copy code app vào container
COPY . /app

# Expose port cho FastAPI
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--env-file", ".env", "--reload"]
