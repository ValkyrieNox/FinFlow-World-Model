param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$DataDir = '',
    [string]$CheckpointRoot = '',
    [string]$OutDir = '',
    [string]$Python = 'python',
    [int]$NPaths = 10000,
    [int]$NSteps = 252,
    [string]$Device = 'auto',
    [int]$FmNSteps = 20,
    [ValidateSet('euler', 'heun')]
    [string]$FmSolver = 'euler',
    [int]$SignatureDepth = 3,
    [int]$Limit = 0,
    [switch]$CalibrateMoments,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $RepoRoot 'data\heston_v3'
}
if ([string]::IsNullOrWhiteSpace($CheckpointRoot)) {
    $CheckpointRoot = Join-Path $RepoRoot 'release\checkpoints'
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot 'runs\full_surface_eval'
}

$real = Join-Path $DataDir 'test.npz'
$oracle = Join-Path $DataDir 'mc_oracle.npz'
$metadata = Join-Path $DataDir 'metadata.json'

foreach ($required in @($RepoRoot, $DataDir, $CheckpointRoot, $real, $oracle, $metadata)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required path: $required"
    }
}

$rolloutScript = Join-Path $RepoRoot 'scripts\rollout_joint.py'
$evalScript = Join-Path $RepoRoot 'scripts\evaluate_rollout.py'
foreach ($required in @($rolloutScript, $evalScript)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required script: $required"
    }
}

$rolloutDir = Join-Path $OutDir 'rollouts'
$evalDir = Join-Path $OutDir 'evals'

function Get-RelativePathSafe([string]$BasePath, [string]$TargetPath) {
    $baseFull = (Resolve-Path -LiteralPath $BasePath).Path.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $targetFull = (Resolve-Path -LiteralPath $TargetPath).Path
    $baseUri = New-Object System.Uri($baseFull)
    $targetUri = New-Object System.Uri($targetFull)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', [IO.Path]::DirectorySeparatorChar)
}
New-Item -ItemType Directory -Force -Path $rolloutDir, $evalDir | Out-Null

# Wide vanilla/Asian moneyness grid. Strikes are K = moneyness * S0.
$moneynesses = @(
    '0.50','0.60','0.70','0.80','0.85','0.90','0.95',
    '1.00','1.05','1.10','1.15','1.20','1.30','1.40','1.50','1.75','2.00'
)
$maturities = @('0.25','0.5','1.0')
$asianMoneynesses = $moneynesses
$asianMaturities = $maturities

$nestedPrefix = (Join-Path $CheckpointRoot 'checkpoints')
$checkpoints = Get-ChildItem -LiteralPath $CheckpointRoot -Recurse -File -Filter 'best.pt' |
    Where-Object { -not $_.FullName.StartsWith($nestedPrefix, [System.StringComparison]::OrdinalIgnoreCase) } |
    Sort-Object FullName

if ($checkpoints.Count -eq 0) {
    throw "No best.pt checkpoints found under $CheckpointRoot"
}

Write-Host "RepoRoot       : $RepoRoot"
Write-Host "DataDir        : $DataDir"
Write-Host "CheckpointRoot : $CheckpointRoot"
Write-Host "OutDir         : $OutDir"
Write-Host "N checkpoints  : $($checkpoints.Count)"
Write-Host "Moneynesses    : $($moneynesses -join ' ')"
Write-Host "Maturities     : $($maturities -join ' ')"

$rows = New-Object System.Collections.Generic.List[object]
$idx = 0
foreach ($ckpt in $checkpoints) {
    $idx += 1
    $rel = Get-RelativePathSafe $CheckpointRoot $ckpt.DirectoryName
    $safe = ($rel -replace '[\\/:*?"<>| ]+', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($safe)) { $safe = "checkpoint_$idx" }

    $rollout = Join-Path $rolloutDir "$safe.npz"
    $evalJson = Join-Path $evalDir "$safe.json"

    Write-Host "[$idx/$($checkpoints.Count)] Rollout: $rel"
    if ($Force -or -not (Test-Path -LiteralPath $rollout)) {
        $rolloutArgs = @(
            $rolloutScript,
            '--checkpoint', $ckpt.FullName,
            '--data-dir', $DataDir,
            '--output', $rollout,
            '--n-paths', [string]$NPaths,
            '--n-steps', [string]$NSteps,
            '--regime-actions',
            '--action-seed', '20260701',
            '--noise-seed', '20260701',
            '--fm-n-steps', [string]$FmNSteps,
            '--fm-solver', $FmSolver,
            '--device', $Device
        )
        if ($CalibrateMoments) { $rolloutArgs += '--calibrate-moments' }
        & $Python @rolloutArgs
        if ($LASTEXITCODE -ne 0) { throw "rollout failed for $($ckpt.FullName)" }
    } else {
        Write-Host "  existing rollout found; use -Force to regenerate"
    }

    Write-Host "[$idx/$($checkpoints.Count)] Evaluate: $rel"
    if ($Force -or -not (Test-Path -LiteralPath $evalJson)) {
        $evalArgs = @(
            $evalScript,
            '--real', $real,
            '--fake', $rollout,
            '--data-dir', $DataDir,
            '--mc-oracle', $oracle,
            '--output', $evalJson,
            '--moneynesses'
        ) + $moneynesses + @(
            '--maturities'
        ) + $maturities + @(
            '--asian-moneynesses'
        ) + $asianMoneynesses + @(
            '--asian-maturities'
        ) + $asianMaturities + @(
            '--signature-depth', [string]$SignatureDepth
        )
        if ($Limit -gt 0) { $evalArgs += @('--limit', [string]$Limit) }
        & $Python @evalArgs | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "evaluation failed for $($ckpt.FullName)" }
    } else {
        Write-Host "  existing evaluation found; use -Force to recompute"
    }

    $report = Get-Content -LiteralPath $evalJson -Raw | ConvertFrom-Json
    $pricing = $report.pricing_fake_vs_mc_oracle
    $asian = $report.asian_pricing_fake_vs_mc_oracle
    $dist = $report.distances
    $sigMean = $null
    if ($null -ne $dist.signature_wasserstein) { $sigMean = $dist.signature_wasserstein.mean }

    $rows.Add([pscustomobject]@{
        model = $rel
        checkpoint = $ckpt.FullName
        rollout = $rollout
        eval_json = $evalJson
        vanilla_rmse = $pricing.rmse_overall
        vanilla_mape = $pricing.mape_overall
        asian_rmse = $asian.rmse_overall
        asian_mape = $asian.mape_overall
        marginal_w1_mean = $dist.marginal_wasserstein_mean
        marginal_w1_max = $dist.marginal_wasserstein_max
        total_return_w1 = $dist.total_return_wasserstein
        abs_total_return_w1 = $dist.abs_total_return_wasserstein
        sig_w1_mean = $sigMean
    }) | Out-Null
}

$summaryCsv = Join-Path $OutDir 'summary_full_surface.csv'
$summaryJson = Join-Path $OutDir 'summary_full_surface.json'
$rows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding UTF8
$rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryJson -Encoding UTF8

Write-Host "Done."
Write-Host "Summary CSV : $summaryCsv"
Write-Host "Summary JSON: $summaryJson"

