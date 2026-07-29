param(
    [string]$OutputDir = "",
    [double]$Threshold = 40129,
    [int]$Positions = 21,
    [int]$MaxStruts = 0,
    [switch]$Overwrite
)

$projectRoot = $PSScriptRoot
$localPackages = Join-Path $projectRoot ".python_packages"
$sourceDir = Join-Path $projectRoot "src"
$tiffPath = Join-Path $projectRoot "data\missing_struts\tif_stacks\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
$jsonPath = Join-Path $projectRoot "data\missing_struts\registered_jsons\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
$thresholdsPath = Join-Path $projectRoot "configs\thin_thick_bent_thresholds.json"
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "data\missing_struts\analysis\thin_thick_bent"
}

$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path -LiteralPath $bundledPython) {
    $pythonPath = $bundledPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Install Python and the packages in requirements.txt."
    }
    $pythonPath = $pythonCommand.Source
}

$env:PYTHONPATH = "$localPackages;$sourceDir"
$arguments = @(
    (Join-Path $sourceDir "strut_defect_pipeline.py"),
    $tiffPath,
    $jsonPath,
    $OutputDir,
    "--threshold", "$Threshold",
    "--thresholds-json", $thresholdsPath,
    "--positions", "$Positions"
)
if ($MaxStruts -gt 0) {
    $arguments += @("--max-struts", "$MaxStruts")
}
if ($Overwrite) {
    $arguments += "--overwrite"
}

& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Thin/thick/bent pipeline failed with exit code $LASTEXITCODE."
}
