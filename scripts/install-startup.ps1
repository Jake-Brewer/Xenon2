# Makes Xenon2 start at login: puts a shortcut to scripts\start-xenon.ps1 in this user's Startup
# folder (no admin needed), and writes the settings file it reads if there isn't one yet.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1            # install
#   powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -Remove    # undo

param([switch]$Remove)

$root = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath('Startup')
$link = Join-Path $startup 'Xenon2.lnk'

if ($Remove) {
    if (Test-Path $link) { Remove-Item $link; "removed $link" } else { "no startup shortcut to remove" }
    exit 0
}

$settings = Join-Path $env:APPDATA 'com.xenon2.app\startup-env.txt'
if (-not (Test-Path $settings)) {
    New-Item -ItemType Directory -Force (Split-Path $settings) | Out-Null
    @(
        '# Settings scripts\start-xenon.ps1 applies when Xenon2 starts at login. KEY=VALUE per line.',
        'XENON_MODEL=rwkv-7-g1l-2.9B-Q5_1.bin',
        '# Heat pacing (see scripts/pace_sweep.py); leave out to use the app defaults.',
        '# XENON_PACE_FROM_C=74',
        '# XENON_PACE_MS_PER_C=12'
    ) | Set-Content $settings -Encoding utf8
    "wrote $settings"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = (Get-Command powershell.exe).Source
$shortcut.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$(Join-Path $root 'scripts\start-xenon.ps1')`""
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'Start Xenon2 at login'
$shortcut.Save()
"installed $link"
