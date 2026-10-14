# Launch training runs from a jobs file, at most -MaxParallel at a time (0 = all at once).
# Each line of the jobs file is the argument list for `python -m scripts.train`.
# Logs go to runs/logs/<run-name>.log.
#
# Usage: powershell -File experiments/run_parallel.ps1 experiments/ppo_vs_dqn.txt -MaxParallel 1

param(
    [Parameter(Mandatory = $true)][string]$JobsFile,
    [int]$MaxParallel = 0
)

$ErrorActionPreference = "Stop"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force runs/logs | Out-Null

$procs = @()
foreach ($line in Get-Content $JobsFile) {
    $line = $line.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { continue }
    while ($MaxParallel -gt 0 -and @($procs | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) {
        Start-Sleep -Seconds 5
    }
    $name = [regex]::Match($line, "--run-name\s+(\S+)").Groups[1].Value
    Write-Host "$(Get-Date -Format HH:mm:ss) start $name"
    $procs += Start-Process python -ArgumentList "-m scripts.train $line" -NoNewWindow -PassThru `
        -RedirectStandardOutput "runs/logs/$name.log" -RedirectStandardError "runs/logs/$name.err"
}
$procs | Wait-Process
foreach ($p in $procs) { Write-Host "pid $($p.Id) exit $($p.ExitCode)" }
