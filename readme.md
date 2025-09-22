1. Các bước cài đặt:
  + Clone : git clone https://github.com/nghiavt64582/AiVoiceRecorder.git
  + Vào folder setup và chạy các file tương ứng
  + Dùng đúng python 3.12 ()
    + Trên Linux: 
      + sudo apt update
      + sudo apt install software-properties-common -y
      + sudo add-apt-repository ppa:deadsnakes/ppa -y
      + sudo apt update
      + sudo apt install python3.12 python3.12-venv python3.12-dev -y
    + Trên Windows:
      + 
  + Cài path cho ffmpeg: Desktop\ffmpeg-2025-09-08-git-45db6945e9-essentials_build\bin
  + Tạo môi trường ảo: 
    + Window : py -3.12 -m venv .venv
    + MacOS/Linux: python3.12 -m venv .venv
  + Kích hoạt virtual environment:
    + Window: .\.venv\Scripts\activate
    + MacOS/Linux: source .venv/bin/activate
    + Thoát khỏi môi trường ảo: deactivate
  + Cài đặt các thư viện cần thiết: pip install -r requirements.txt
  + Cài torch cho gpu : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  + pip install fastapi huggingface_hub openai_whisper pydantic requests transformers whisperx whisper uvicorn dotenv accelerate transformers fastapi openai-whisper python-multipart einops transformers_stream_generator
  + Chạy ứng dụng: 
    + Window: uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ..\.env --reload
    + Linux: uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ../.env --reload
  + Xem nvidia info: nvidia-smi
  + Forward port bằng cloudflare để test từ máy gốc
    + Vào setup mở cloudflare chạy tunnel --url http://localhost:8000

==> bat để chạy trên linux: run.sh
      + sudo apt update
      + sudo apt install software-properties-common -y
      + sudo add-apt-repository ppa:deadsnakes/ppa -y
      + sudo apt update
      + sudo apt install python3.12 python3.12-venv python3.12-dev -y
      + python3.12 -m venv .venv
      + source .venv/bin/activate
      + apt install ffmpeg -y
      + pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      + pip install fastapi huggingface_hub openai_whisper pydantic requests transformers whisperx whisper uvicorn dotenv accelerate transformers fastapi openai-whisper python-multipart einops transformers_stream_generator
      + cd src
      + uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ../.env --reload


2. Forward port để test bằng cloud flare:
  + Tải cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
  + Giải nén và cài đặt cloudflared
  + Vào thư mục chứa cloudflared chạy : tunnel --url http://localhost:8000

3. Đối với whisperx:
  + Tạo token trên huggingface và set biến môi trường HF_TOKEN để dùng diarization
  + Cần accept điều khoản sử dụng model whisperx trên huggingface

4. Tip:
  + Dùng pip freeze > requirements.txt để lưu các thư viện trong môi trường ảo
  + Untrack các file .pyc trong git bằng cách thêm vào .gitignore file:
    + *.pyc
    + __pycache__/