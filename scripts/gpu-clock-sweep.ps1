# Tries several GPU core-clock caps and measures Xenon's speed at each, then resets the clocks.
# Run from an ADMIN PowerShell (locking clocks needs it), with the Xenon2 app running:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gpu-clock-sweep.ps1
#
# Writing tokens is limited by memory speed, not by the core clock, so a lower core clock should
# cost little speed and much less heat. Only the core (graphics) clock is capped; memory stays at
# full speed. Results go to training\out\gpu-clock-sweep.txt (UTF-8). The clocks are reset at the
# end (nvidia-smi -rgc), and a reboot resets them too.

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'voice-pipeline\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$out = Join-Path $root 'training\out\gpu-clock-sweep.txt'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

function Note($line) {
    Write-Host $line
    # Out-File, not Tee-Object: Windows PowerShell's Tee-Object writes UTF-16.
    $line | Out-File -FilePath $out -Append -Encoding utf8
}

"GPU clock sweep $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File -FilePath $out -Encoding utf8
Write-Host (Get-Content $out)

try {
    foreach ($cap in 'none', 1500, 1200, 900, 'none') {
        if ($cap -eq 'none') {
            nvidia-smi -rgc | Out-Null
        } else {
            nvidia-smi -lgc "300,$cap" | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Note "cap $cap`: nvidia-smi -lgc failed (needs an admin PowerShell); skipped"
                continue
            }
        }
        # One measurement may fail (the app dropping a connection, say) without ending the sweep:
        # try twice, then note the failure and move on. Python's own error output is captured,
        # since Windows PowerShell 5.1 can turn a native program's stderr into a fatal error.
        $env:PYTHONIOENCODING = 'utf-8'
        $result = $null
        foreach ($try in 1, 2) {
            $lines = & $python (Join-Path $root 'scripts\gpu_speed.py') --label "$cap" 2>&1 | ForEach-Object { "$_" }
            if ($LASTEXITCODE -eq 0) { $result = $lines | Select-Object -Last 1; break }
            Write-Host "cap $cap, try $try failed: $(($lines | Select-Object -Last 1))"
        }
        if ($result) { Note $result } else { Note "cap $cap`: measurement failed twice" }
    }
} finally {
    nvidia-smi -rgc | Out-Null
    Note "clocks reset"
}
