@echo off
echo ================================================
echo Medical AI Backend - Starting
echo ================================================

cd /d "D:\Final Year Project\backend"

echo.
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/2] Starting backend...
echo.
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Note: No Ollama needed! Using Groq cloud API
echo.
echo Press Ctrl+C to stop
echo.

python main.py

pause