@echo off
echo ==================================================
echo   ASISTEN DIGITAL PENGASUHAN - POLTEK SSN
echo ==================================================
echo.
echo Menutup server lama (jika ada)...
taskkill /F /IM python.exe /T >nul 2>&1

echo Sedang menyiapkan server... mohon tunggu sebentar.
start /b streamlit run app.py --server.port 8501 --server.headless true

echo Membuka Aplikasi...
timeout /t 5 >nul
start http://localhost:8501

echo.
echo [INFO] Sistem sudah berjalan di http://localhost:8501
echo [INFO] Tekan CTRL+C untuk mematikan server.
echo.
pause
