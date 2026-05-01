@echo off
echo ==============================================
echo   Video Enhancement Server
echo ==============================================
echo.

REM Set environment variables
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

REM Activate virtual environment if exists
if exist "gemma4-edge\Scripts\activate.bat" (
    call gemma4-edge\Scripts\activate.bat
    echo Virtual environment activated
)

echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop
echo ==============================================
echo.

python app.py

pause