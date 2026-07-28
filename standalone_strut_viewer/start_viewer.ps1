param(
    [int]$Port = 8780
)

$viewerRoot = $PSScriptRoot
$localPackages = Join-Path $viewerRoot ".python_packages"
$projectPackages = Join-Path (Split-Path $viewerRoot -Parent) ".python_packages"
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path -LiteralPath $bundledPython) {
    $pythonPath = $bundledPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Install Python 3.10+ and run: pip install -r requirements.txt"
    }
    $pythonPath = $pythonCommand.Source
}

$packageRoots = @($localPackages, $projectPackages) |
    Where-Object { Test-Path -LiteralPath $_ }
if ($packageRoots.Count) {
    $env:PYTHONPATH = ($packageRoots -join ";")
}

& $pythonPath -c "import numpy, scipy, tifffile" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Missing Python packages. From this folder run: python -m pip install -r requirements.txt"
}

Write-Host "Opening the standalone strut viewer at http://127.0.0.1:$Port/"
Write-Host "Upload a TIFF, registered JSON, and flagged-strut CSV in the browser."
Write-Host "Press Ctrl+C to stop; temporary TIFF data will be deleted."

& $pythonPath (Join-Path $viewerRoot "server.py") --port $Port
