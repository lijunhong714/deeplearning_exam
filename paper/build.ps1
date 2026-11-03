$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ((Test-Path -LiteralPath '..\outputs\paper_main\summary.csv') -or (Test-Path -LiteralPath '..\experiment_records\paper_main\summary.csv')) {
        python build_results.py
        if ($LASTEXITCODE -ne 0) { throw 'Result validation failed.' }
    } else {
        Write-Host 'Using the included result tables; experiment records are not present.'
    }
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'First XeLaTeX pass failed.' }
    bibtex main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Second XeLaTeX pass failed.' }
    xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Final XeLaTeX pass failed.' }
} finally {
    Pop-Location
}
