$ErrorActionPreference = 'Stop'

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'Missing .venv. Run .\setup.ps1 first.'
}

$steps = @(
    'v3_NB1_EDA.py',
    'v3_NB2_KnowledgeLayer.py',
    'v3_NB3_Training.py',
    'test_harness.py',
    'v3_NB6_Evaluation.py'
)

foreach ($step in $steps) {
    Write-Host "Running $step..."
    & $venvPython (Join-Path $PSScriptRoot $step)
}

Write-Host 'Project flow complete.'
Write-Host 'Run v3_NB4_Pipeline.py separately for the interactive recommendation flow.'