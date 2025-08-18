1. Các lệnh:
  + Tạo môi trường ảo: python -m venv venv
  + Kích hoạt virtual environment:
    + Window: .\.venv\Scripts\activate
    + MacOS/Linux: source .venv/bin/activate
  + Cài đặt các thư viện cần thiết: pip install -r requirements.txt
  + Chạy ứng dụng: uvicorn rest:app --reload