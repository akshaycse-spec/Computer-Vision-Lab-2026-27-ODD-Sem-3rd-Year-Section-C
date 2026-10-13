@echo off
setlocal
cd /d C:\BTCS301P-AS01

echo ==============================================================
echo BTCS301P-AS01 - Setup and Run
echo ==============================================================

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)

%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Make sure Python and pip are installed.
    pause
    exit /b 1
)

echo.
echo Running assignment...
%PYTHON% main.py --show

echo.
echo Finished. Check C:\BTCS301P-AS01\output for all generated files.
pause
