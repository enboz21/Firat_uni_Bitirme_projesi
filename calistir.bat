@echo off
setlocal

cd /d "%~dp0"

if not exist "venv" (
    echo Sanal ortam olusturuluyor...
    python -m venv venv
    call venv\Scripts\activate
    echo Kutuphaneler yukleniyor...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

python gol_pixel_sayici.py %*

call venv\Scripts\deactivate.bat
endlocal
