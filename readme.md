1. Các lệnh:
  + Dùng đúng python 3.12 ()
  + Tạo môi trường ảo: python -m venv .venv
  + Kích hoạt virtual environment:
    + Window: .\.venv\Scripts\activate
    + MacOS/Linux: source .venv/bin/activate
  + pip install accelerate transformers fastapi uvicorn openai-whisper python-multipart einops transformers_stream_generator hf_xet
  + Cài torch bằng pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  + Cài đặt các thư viện cần thiết: pip install -r requirements.txt
  + Chạy ứng dụng: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  + Xem nvidia info: nvidia-smi
  + Cài đặt thêm ffmpeg: https://ffmpeg.org/download.html