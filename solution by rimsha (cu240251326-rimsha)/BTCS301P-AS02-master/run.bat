@echo off
setlocal
cd /d "%~dp0"

echo =============================================
echo BTCS301P-AS02 - Experiment No. 2
echo =============================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher ^(py^) was not found.
    echo Install Python 3.10+ from python.org and enable Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"

echo Installing/updating required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Running experiment...
python main.py
if errorlevel 1 goto :error

echo.
echo Completed successfully.
echo Results are available in: %~dp0output
pause
exit /b 0

:error
echo.
echo An error occurred. Review the message above.
pause
exit /b 1
