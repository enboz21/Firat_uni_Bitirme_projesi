@echo off
setlocal

cd /d "%~dp0"

if not exist "venvgpu" (
    echo Sanal ortam olusturuluyor...
    python -m venv venvgpu
    call venvgpu\Scripts\activate
    echo Ozel Torch versiyonu yukleniyor...
    pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    echo Diger kutuphaneler yukleniyor...
    pip install -r requirementsgpu.txt
) else (
    call venvgpu\Scripts\activate
)

python gol_pixel_sayici.py %*

call venvgpu\Scripts\deactivate.bat
endlocal
