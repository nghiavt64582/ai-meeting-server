# Base image Python 3.12 với slim Debian (nhẹ)
FROM python:3.12-slim

# Đặt thư mục làm việc
WORKDIR /app

# --- Cài gói hệ thống cần thiết ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Cài PyTorch GPU ---
# Lưu ý: thay cu121 bằng phiên bản CUDA phù hợp với máy (VD: cu126, cu128...)
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
RUN apt-get update && apt-get install -y patchelf && rm -rf /var/lib/apt/lists/*

# Quét toàn bộ site-packages thay vì .venv
RUN find /usr/local/lib/python3.12/site-packages -name "libctranslate2*.so*" \
    -exec patchelf --clear-execstack {} \; || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy toàn bộ code app vào container
COPY . /app

# Expose port cho FastAPI
EXPOSE 8000

# Lệnh chạy app
CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--env-file", ".env", "--reload"]
