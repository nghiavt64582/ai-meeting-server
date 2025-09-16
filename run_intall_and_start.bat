@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: =======================
::  CONFIG & PATHS
:: =======================
set "ROOT=%~dp0"
set "SETUP_DIR=%ROOT%setup"
set "SRC_DIR=%ROOT%src"
set "ENV_FILE=%ROOT%.env"
set "VENV_DIR=%ROOT%.venv"
set "PY_SETUP=%SETUP_DIR%\python-3.12.10-amd64.exe"
set "CLOUDFLARED_EXE=%SETUP_DIR%\cloudflared-windows-amd64.exe"

echo.
echo [*] Project root  : %ROOT%
echo [*] Setup folder  : %SETUP_DIR%
echo [*] Source folder : %SRC_DIR%
echo [*] .env file     : %ENV_FILE%

:: =======================
::  FIND FFMPEG BIN FOLDER
:: =======================
set "FFMPEG_BIN="
for /d %%D in ("%SETUP_DIR%\ffmpeg-*") do (
    if exist "%%~fD\bin\ffmpeg.exe" (
        set "FFMPEG_BIN=%%~fD\bin"
        goto :FFOK
    )
)
:FFOK
if "%FFMPEG_BIN%"=="" (
    echo [!] Khong tim thay ffmpeg trong ^"%SETUP_DIR%\ffmpeg-*\bin^"
    echo     Bo qua buoc them PATH cho ffmpeg. (Dam bao FFmpeg co trong PATH)
) else (
    echo [*] FFmpeg bin   : %FFMPEG_BIN%
    :: Thêm FFmpeg vào PATH (session)
    set "PATH=%FFMPEG_BIN%;%PATH%"
    :: Và persist vào PATH (User) nếu chưa có
    powershell -NoProfile -Command ^
      "$p=[Environment]::GetEnvironmentVariable('Path','User');" ^
      "if(-not $p.ToLower().Contains('%FFMPEG_BIN%'.ToLower())){[Environment]::SetEnvironmentVariable('Path',$p+';%FFMPEG_BIN%','User'); Write-Host '   [+] Added ffmpeg to User PATH'} else {Write-Host '   [=] ffmpeg already in User PATH'}"
)

:: =======================
::  INSTALL PYTHON 3.12 (silent) NẾU CHƯA CÓ
:: =======================
where py >nul 2>&1
if %errorlevel% neq 0 (
    where python >nul 2>&1
)
if %errorlevel% neq 0 (
    if not exist "%PY_SETUP%" (
        echo [x] Khong tim thay installer Python: %PY_SETUP%
        echo     Dat file python-3.12.10-amd64.exe trong thu muc ^"setup^" roi chay lai.
        pause & exit /b 1
    )
    echo [*] Dang cai Python 3.12 (silent)...
    "%PY_SETUP%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 SimpleInstall=1
    if %errorlevel% neq 0 (
        echo [x] Cai dat Python that bai.
        pause & exit /b 1
    )
) else (
    echo [*] Python da co san.
)

:: =======================
::  TẠO VENV VÀ CÀI LIBS
:: =======================
:: Dùng py -3.12 nếu có, fallback sang python
set "PY_CMD=py -3.12"
%PY_CMD% -V >nul 2>&1
if %errorlevel% neq 0 (
    set "PY_CMD=python"
)

if not exist "%VENV_DIR%" (
    echo [*] Tao virtualenv: %VENV_DIR%
    %PY_CMD% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [x] Tao venv that bai.
        pause & exit /b 1
    )
) else (
    echo [*] Virtualenv da ton tai.
)

call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [x] Khong kich hoat duoc venv.
    pause & exit /b 1
)

python -m pip install -U pip wheel
if %errorlevel% neq 0 (
    echo [x] Loi khi nang cap pip/wheel.
    pause & exit /b 1
)

echo [*] Cai cac thu vien chinh...
pip install ^
 accelerate transformers fastapi uvicorn openai-whisper ^
 python-multipart einops transformers_stream_generator whisperx optimum
if %errorlevel% neq 0 (
    echo [!] Canh bao: co the co goi nao cai dat chua thanh cong. Tiep tuc...
)

:: =======================
::  TORCH CHO GPU (nếu có NVIDIA)
:: =======================
set "NVSMI=%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
if exist "%NVSMI%" (
    echo [*] NVIDIA GPU phat hien -> Cai torch CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo [!] Khong cai duoc torch GPU. Tiep tuc voi CPU...
    )
) else (
    echo [=] Khong thay NVIDIA GPU -> bo qua cai torch GPU.
)

:: =======================
::  CAI REQUIREMENTS.TXT (neu co)
:: =======================
if exist "%ROOT%requirements.txt" (
    echo [*] pip install -r requirements.txt
    pip install -r "%ROOT%requirements.txt"
) else (
    echo [=] Khong co requirements.txt o root.
)

:: =======================
::  CHẠY UVICORN (start cửa sổ mới)
:: =======================
if not exist "%SRC_DIR%\main.py" (
    echo [x] Khong tim thay main.py trong: %SRC_DIR%
    pause & exit /b 1
)

pushd "%SRC_DIR%"
set "ENV_ARG="
if exist "%ENV_FILE%" (
    set "ENV_ARG=--env-file ..\.env"
)
echo [*] Start Uvicorn o cua so moi...
start "Uvicorn API" cmd /k ".\..\ .venv\Scripts\activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload %ENV_ARG%"
popd

:: =======================
::  CHẠY CLOUDFLARE TUNNEL (start cửa sổ mới)
:: =======================
if exist "%CLOUDFLARED_EXE%" (
    echo [*] Start Cloudflared tunnel o cua so moi...
    start "Cloudflared" "%CLOUDFLARED_EXE%" tunnel --url http://localhost:8000
) else (
    echo [!] Khong tim thay cloudflared: %CLOUDFLARED_EXE%
)

echo.
echo [+] DONE. Uvicorn va Cloudflared da duoc mo o cua so rieng.
echo     Neu thay PATH chua nhan FFmpeg, hay mo terminal moi hoac dang xuat/dang nhap lai.
echo.
pause
endlocal
