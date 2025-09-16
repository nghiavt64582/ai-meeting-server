1. Các bước cài đặt:
  + Vào folder setup và chạy các file tương ứng
  + Dùng đúng python 3.12 ()
  + Cài path cho ffmpeg: Desktop\ffmpeg-2025-09-08-git-45db6945e9-essentials_build\bin
  + Tạo môi trường ảo: py -3.12 -m venv .venv
  + Kích hoạt virtual environment:
    + Window: .\.venv\Scripts\activate
    + MacOS/Linux: source .venv/bin/activate
  + pip install accelerate transformers fastapi uvicorn openai-whisper python-multipart einops transformers_stream_generator whisperx optimum
  + Cài torch cho gpu : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  + Cài đặt các thư viện cần thiết: pip install -r requirements.txt
  + Chạy ứng dụng: uvicorn main:app --host 0.0.0.0 --port 8000 --env-file ..\.env --reload
  + Xem nvidia info: nvidia-smi
  + Forward port bằng cloudflare để test từ máy gốc
    + Vào setup mở cloudflare chạy tunnel --url http://localhost:8000

2. Forward port để test bằng cloud flare:
  + Tải cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
  + Giải nén và cài đặt cloudflared
  + Vào thư mục chứa cloudflared chạy : tunnel --url http://localhost:8000

3. Đối với whisperx:
  + Tạo token trên huggingface và set biến môi trường HF_TOKEN để dùng diarization
  + Cần accept điều khoản sử dụng model whisperx trên huggingface

4. Tip:
  + Dùng pip freeze > requirements.txt để lưu các thư viện trong môi trường ảo