@echo off
title Real-ESRGAN Studio
echo.
echo  ========================================
echo    Real-ESRGAN Super Resolution Studio
echo  ========================================
echo.
echo  Starting app... Please wait.
echo.

cd /d "%~dp0"

:: Launch the app in background and open browser after 4 seconds
start "" /B python app.py

:: Wait for Gradio to spin up then open browser
timeout /t 4 /nobreak >nul
start "" "http://localhost:7860"

:: Keep the window open to show logs
python app.py
pause
