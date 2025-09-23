FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install git+https://github.com/m-bain/whisperx.git@v3.4.2 \
    && pip install accelerate transformers fastapi uvicorn \
       python-multipart einops transformers_stream_generator python-dotenv
       
RUN apt-get update && apt-get install -y gcc curl \
    && curl -L -o /tmp/execstack.c https://raw.githubusercontent.com/hjl-tools/prelink/master/execstack.c \
    && gcc /tmp/execstack.c -o /usr/local/bin/execstack \
    && rm /tmp/execstack.c \
    && execstack -c /usr/local/lib/python3.12/site-packages/ctranslate2/*.so || true \
    && execstack -c /usr/local/lib/python3.12/site-packages/ctranslate2/_ext/*.so || true \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--env-file", "../.env"]