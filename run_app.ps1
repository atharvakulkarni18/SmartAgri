$ErrorActionPreference = 'Stop'

$setupScript = Join-Path $PSScriptRoot 'setup.ps1'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    & $setupScript
}

if (-not (Test-Path $venvPython)) {
    throw 'Virtual environment was not created. Run .\setup.ps1 and try again.'
}

& $venvPython -m streamlit run (Join-Path $PSScriptRoot 'app.py')