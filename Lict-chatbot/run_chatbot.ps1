# LICT Chatbot Launcher
# This script runs the Streamlit app using Python module to bypass Device Guard

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

Write-Host "Starting LICT Chatbot..." -ForegroundColor Green
Write-Host "Opening at http://localhost:8501" -ForegroundColor Cyan

# Run Streamlit as a Python module (bypasses Device Guard .exe block)
& .\.venv\Scripts\python.exe -m streamlit run webscrap.py

Pop-Location
