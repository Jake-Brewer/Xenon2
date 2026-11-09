# Starts the Xenon2 release app with its saved settings. Run at login by the Startup-folder
# shortcut that scripts\install-startup.ps1 makes, so Xenon comes back after a reboot.
#
# Settings: %APPDATA%\com.xenon2.app\startup-env.txt, one KEY=VALUE per line (lines starting with
# # are comments), e.g. XENON_MODEL, XENON_PACE_FROM_C, XENON_PACE_MS_PER_C. Missing file: the model
# defaults to G1l 2.9B and pacing to the app's defaults.
#
# Does nothing if the app is already running. Waits (up to 5 minutes) for the GPU driver after a
# reboot, so the model loads on the GPU rather than falling back to the CPU. Logs to
# %TEMP%\xenon2-release.log, as scripts/run-release.sh does.

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'app\src-tauri\target\release\app.exe'
$log = Join-Path $env:TEMP 'xenon2-release.log'
$settings = Join-Path $env:APPDATA 'com.xenon2.app\startup-env.txt'

if (-not (Test-Path $exe)) {
    "$(Get-Date -Format s) start-xenon: no release build at $exe (run scripts/run-release.sh once)" | Out-File $log -Append -Encoding utf8
    exit 1
}
if (Get-Process app -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }) {
    exit 0
}

$env:XENON_MODEL = 'rwkv-7-g1l-2.9B-Q5_1.bin'
if (Test-Path $settings) {
    foreach ($line in Get-Content $settings) {
        $line = $line.Trim()
        if ($line -eq '' -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
        $name, $value = $line.Split('=', 2)
        Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
    }
}

# Right after login the NVIDIA driver can lag; loading then would put the model on the CPU.
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    nvidia-smi -L *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 5
}

Start-Process -FilePath $exe -WorkingDirectory $root -WindowStyle Normal `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

# Tell the other GPU user this app holds ~3 GB while open (see AGENTS.md's GPU rules).
$lease = Join-Path $env:USERPROFILE '.gpu\LEASE.txt'
if (Test-Path $lease) {
    $lines = Get-Content $lease | Where-Object { $_ -notmatch '^xenon2-45 ' }
    $lines += "xenon2-45 | $(Get-Date -Format 'yyyy-MM-dd HH:mm') | while the Xenon2 app is open | ~3.0 GB resident | shared"
    $lines | Set-Content $lease -Encoding utf8
}
