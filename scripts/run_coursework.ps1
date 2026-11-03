$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'run_experiments.py') @args
exit $LASTEXITCODE
