FROM lscr.io/linuxserver/faster-whisper:gpu

ENV TZ=Asia/Bangkok PUID=1000 PGID=1000 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Tools cho app
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash python3 python3-venv \
    ffmpeg libsndfile1 sed \
 && rm -rf /var/lib/apt/lists/*

# Venv RIÊNG cho app (không đổi PATH toàn cục)
RUN python3 -m venv /app/.venv

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Cài deps vào venv (dùng pip của venv)
RUN /app/.venv/bin/pip install --upgrade pip wheel setuptools && \
    /app/.venv/bin/pip install -r /app/requirements.txt && \
    /app/.venv/bin/pip install --no-deps git+https://github.com/m-bain/whisperx.git@v3.4.2 && \
    /app/.venv/bin/pip install accelerate transformers uvicorn fastapi python-multipart einops transformers_stream_generator python-dotenv

# Cài torch riêng hỗ trợ cache layer
RUN /app/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
      torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --no-cache-dir

# Mã nguồn
COPY . /app

# Service s6 cho app của bạn
RUN mkdir -p /etc/services.d/app
RUN cat > /etc/services.d/app/run <<'EOF'
#!/command/with-contenv /bin/bash
set -e
cd /app/src
exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ../.env
EOF

# Chuyển CRLF -> LF và cấp quyền thực thi
RUN sed -i 's/\r$//' /etc/services.d/app/run && chmod +x /etc/services.d/app/run

# TẮT service Wyoming mặc định để không spam
RUN mkdir -p /etc/services.d/svc-whisper && touch /etc/services.d/svc-whisper/down

EXPOSE 8000 10300
