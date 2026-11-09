# Tries several GPU core-clock caps and measures Xenon's speed at each, then resets the clocks.
# Run from an ADMIN PowerShell (locking clocks needs it), with the Xenon2 app running:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gpu-clock-sweep.ps1
#
# Writing tokens is limited by memory speed, not by the core clock, so a lower core clock should
# cost little speed and much less heat. Only the core (graphics) clock is capped; memory stays at
# full speed. Results go to training\out\gpu-clock-sweep.txt. The clocks are reset at the end
# (nvidia-smi -rgc), and a reboot resets them too.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'voice-pipeline\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$out = Join-Path $root 'training\out\gpu-clock-sweep.txt'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
"GPU clock sweep $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Tee-Object -FilePath $out

try {
    foreach ($cap in 'none', 1500, 1200, 900, 'none') {
        if ($cap -eq 'none') {
            nvidia-smi -rgc | Out-Null
        } else {
            nvidia-smi -lgc "300,$cap" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "nvidia-smi -lgc failed (run this from an admin PowerShell)" }
        }
        $env:PYTHONIOENCODING = 'utf-8'
        & $python (Join-Path $root 'scripts\gpu_speed.py') --label "$cap" | Tee-Object -FilePath $out -Append
    }
} finally {
    nvidia-smi -rgc | Out-Null
    "clocks reset" | Tee-Object -FilePath $out -Append
}
