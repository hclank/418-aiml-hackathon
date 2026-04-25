param(
    [string]$TeamSlug = "Fantastic-Four",
    [switch]$Force,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$SubmissionsRoot = Split-Path -Parent $Source
$Target = Join-Path $SubmissionsRoot $TeamSlug
$ResolvedRoot = (Resolve-Path $SubmissionsRoot).Path

if ($TeamSlug -notmatch '^[A-Za-z0-9-]+$') {
    throw "TeamSlug must contain only letters, digits, and dashes."
}

if (Test-Path $Target) {
    if (-not $Force) {
        throw "Target already exists: $Target. Re-run with -Force to replace it."
    }
    $ResolvedTarget = (Resolve-Path $Target).Path
    if (-not $ResolvedTarget.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove target outside submissions root: $ResolvedTarget"
    }
    Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force
}

New-Item -ItemType Directory -Force $Target | Out-Null

$RootFiles = @(
    "README.md",
    "requirements.txt",
    "infer.py",
    "train.py",
    "profile_model.py",
    "check_submission.py",
    "prepare_submission.ps1",
    "download_xview3.py",
    "extract_xview3.py",
    "verify_xview3.py"
)

foreach ($File in $RootFiles) {
    $From = Join-Path $Source $File
    if (Test-Path $From) {
        Copy-Item -LiteralPath $From -Destination (Join-Path $Target $File)
    }
}

$Dirs = @("src", "sample_input", "docs", "configs")
foreach ($Dir in $Dirs) {
    $From = Join-Path $Source $Dir
    if (Test-Path $From) {
        Copy-Item -LiteralPath $From -Destination (Join-Path $Target $Dir) -Recurse
    }
}

$DataTarget = Join-Path $Target "data\xview3"
New-Item -ItemType Directory -Force $DataTarget | Out-Null
foreach ($File in @("train.csv", "validation.csv")) {
    $From = Join-Path $Source "data\xview3\$File"
    if (Test-Path $From) {
        Copy-Item -LiteralPath $From -Destination (Join-Path $DataTarget $File)
    }
}

$MetricsSource = Join-Path $Source "artifacts\tiny_improved\metrics.jsonl"
if (Test-Path $MetricsSource) {
    $MetricsTarget = Join-Path $Target "artifacts\tiny_improved"
    New-Item -ItemType Directory -Force $MetricsTarget | Out-Null
    Copy-Item -LiteralPath $MetricsSource -Destination (Join-Path $MetricsTarget "metrics.jsonl")
}

$ArtifactsSource = Join-Path $Source "artifacts"
if (Test-Path $ArtifactsSource) {
    $PreferredRun = $null
    foreach ($RunName in @("terramind_trainval", "terramind_moredata", "terramind_finetuned", "terramind_small")) {
        $CandidateRun = Join-Path $ArtifactsSource $RunName
        if (Test-Path $CandidateRun) {
            $PreferredRun = $CandidateRun
            break
        }
    }
    if ($PreferredRun -and (Test-Path $PreferredRun)) {
        $RunName = Split-Path -Leaf $PreferredRun
        $RunTarget = Join-Path (Join-Path $Target "artifacts") $RunName
        New-Item -ItemType Directory -Force $RunTarget | Out-Null
        foreach ($File in @("best.pt", "metrics.jsonl")) {
            $From = Join-Path $PreferredRun $File
            if (Test-Path $From) {
                Copy-Item -LiteralPath $From -Destination (Join-Path $RunTarget $File)
            }
        }
    }
}

$OutTarget = Join-Path $Target "out"
New-Item -ItemType Directory -Force $OutTarget | Out-Null
foreach ($File in @("tiny_detections.csv", "tiny_detections_improved.csv", "terramind_scene_detections.csv", "terramind_finetuned_scene_detections.csv", "terramind_moredata_scene_detections.csv", "terramind_trainval_scene_detections.csv")) {
    $From = Join-Path $Source "out\$File"
    if (Test-Path $From) {
        Copy-Item -LiteralPath $From -Destination (Join-Path $OutTarget $File)
    }
}

Get-ChildItem -Path $Target -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $Target -Recurse -Directory -Filter ".ipynb_checkpoints" | Remove-Item -Recurse -Force

python -B (Join-Path $Target "check_submission.py") $Target

if ($Zip) {
    $ZipPath = Join-Path $SubmissionsRoot "$TeamSlug.zip"
    if (Test-Path $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Compress-Archive -Path $Target -DestinationPath $ZipPath
    Write-Host "Created $ZipPath"
}

Write-Host "Prepared clean submission folder: $Target"
