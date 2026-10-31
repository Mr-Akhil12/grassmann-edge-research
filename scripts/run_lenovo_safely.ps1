#requires -Version 5.1
<# Guarded Lenovo trials. No admin, driver, clocks, BIOS or Windows changes.
Run: powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode inspect
     powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode p1
     powershell -NoProfile -File scripts\run_lenovo_safely.ps1 -Mode p2
#>
param(
    [ValidateSet("inspect", "p1", "p2")] [string]$Mode = "inspect",
    [int]$MaxTempC = 80,
    [int]$MaxGpuMemoryMiB = 3400,
    [int]$MinFreeRamGiB = 8,
    [int]$MaxTrialSeconds = 240
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "results"
if (-not (Test-Path $python)) { throw "Project .venv missing" }
if (-not (Test-Path $logs)) { New-Item -ItemType Directory $logs | Out-Null }
if ($MaxTempC -gt 80 -or $MaxGpuMemoryMiB -gt 3400 -or
    $MinFreeRamGiB -lt 8 -or $MaxTrialSeconds -gt 240) {
    throw "Cannot relax upper safety bounds"
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "NVIDIA monitor missing"
}
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat -and @($bat | Where-Object { $_.BatteryStatus -in @(1,4,5) }).Count -gt 0) {
    throw "Battery is discharging or critical"
}
function Guard {
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $raw = & nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits
    $parts = (@($raw)[0] -split ",")
    if ($parts.Count -lt 3) { throw "Missing GPU telemetry" }
    $temp = [double]::Parse($parts[0].Trim(), [Globalization.CultureInfo]::InvariantCulture)
    $mem  = [double]::Parse($parts[1].Trim(), [Globalization.CultureInfo]::InvariantCulture)
    if ($temp -ge $MaxTempC -or $mem -ge $MaxGpuMemoryMiB -or $free -lt $MinFreeRamGiB) {
        throw "Safety guard triggered: temp=$temp C, VRAM=$mem MiB, freeRAM=$free GiB"
    }
    Write-Output "Guard OK: temp=$temp C; VRAM=$mem MiB; freeRAM=$([math]::Round($free,2)) GiB"
}
Guard
if ($Mode -eq "inspect") {
    Push-Location $root
    try {
        & $python scripts\host_probe.py
        & $python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Correctness tests failed" }
    } finally { Pop-Location }
    exit 0
}
$jobs = if ($Mode -eq "p1") {
    @(
      @("-m","scripts.controlled_synthetic","--device","cuda","--steps","60","--seeds","11,17,23","--lag","4","--out","results/lenovo_p1_lag4.json"),
      @("-m","scripts.controlled_synthetic","--device","cuda","--steps","60","--seeds","11,17,23","--lag","24","--out","results/lenovo_p1_lag24.json")
    )
} else {
    @(,@("-m","scripts.benchmark_p2","--device","cuda","--lengths","16,32,64,128","--warmups","2","--repeats","5","--out","results/lenovo_p2_gpu.json"))
}
foreach ($trial in $jobs) {
    Guard
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outLog = Join-Path $logs ("guarded-" + $stamp + ".stdout.log")
    $errLog = Join-Path $logs ("guarded-" + $stamp + ".stderr.log")
    $env:OMP_NUM_THREADS = "4"
    $env:MKL_NUM_THREADS = "4"
    $kwargs = @{
        FilePath=$python; ArgumentList=$trial; WorkingDirectory=$root;
        PassThru=$true; NoNewWindow=$true;
        RedirectStandardOutput=$outLog; RedirectStandardError=$errLog
    }
    Write-Output ("Running: " + ($trial -join " "))
    $child = Start-Process @kwargs
    $started = Get-Date
    $reason = $null
    while (-not $child.HasExited) {
        Start-Sleep -Seconds 2
        $child.Refresh()
        try {
            Guard
            if (((Get-Date) - $started).TotalSeconds -gt $MaxTrialSeconds) {
                throw "Run time budget reached"
            }
        } catch {
            $reason = $_.Exception.Message
            if (-not $child.HasExited) { Stop-Process -Id $child.Id -Force }
            break
        }
    }
    $child.WaitForExit()
    Write-Output "Exited $($child.ExitCode); logs: $outLog ; $errLog"
    if (Test-Path $outLog) { Get-Content $outLog -Tail 20 }
    if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
    if ($reason) { throw $reason }
    if ($child.ExitCode -ne 0) { throw "Child trial failed" }
}
Guard
