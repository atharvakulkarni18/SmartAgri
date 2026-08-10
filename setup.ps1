$ErrorActionPreference = 'Stop'

function Get-PythonCommand {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw 'Python was not found. Install Python 3.13 and make sure python or py is available in PATH.'
}

$pythonCommand = Get-PythonCommand
$venvPath = Join-Path $PSScriptRoot '.venv'

if (Test-Path $venvPath) {
    Remove-Item -Recurse -Force $venvPath
}

& $pythonCommand -m venv $venvPath

$venvPython = Join-Path $venvPath 'Scripts\python.exe'
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
& $venvPython -c "import numpy, pandas, sklearn; print('Environment check passed')"
Write-Host 'Setup complete. Activate the environment with: .\.venv\Scripts\Activate.ps1'
Write-Host 'Run the project check with: .\.venv\Scripts\python.exe test_harness.py'