# Run the Streamlit RAG Playground app using the workspace virtual environment.
# Usage:
#   .\run_app.ps1

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error "Python executable not found in virtual environment: $venvPython"
    exit 1
}

& $venvPython -m streamlit run "$PSScriptRoot\app.py"
