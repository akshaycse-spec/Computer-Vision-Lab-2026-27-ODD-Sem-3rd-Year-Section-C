$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "BTCS301P-AS02 - Experiment No. 2"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.10+ and enable Add Python to PATH."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".venv\Scripts\python.exe" main.py

Write-Host "Completed. Results are in $PSScriptRoot\output"
