# Install the Rerun NATIVE viewer on a Windows client for remote review.
#
# Installs uv (if missing), the pinned rerun-sdk as a uv tool (its wheel ships
# the `rerun` viewer binary), and ffmpeg. Idempotent: safe to re-run.
#
# $RerunVersion must equal the server's rerun-sdk (pyproject.toml pins
# rerun-sdk==0.37.1); the viewer and the gRPC proxy must be the same version.
#
# ffmpeg: Rerun's native viewer does not bundle a decoder. H.264/H.265/VP8/VP9
# are decoded through a separately installed `ffmpeg` executable found on PATH,
# minimum version 5.1 (https://rerun.io/docs/concepts/logging-and-ingestion/video).
#
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1 [-ServerHost host.example.ts.net]
# Env:   $env:SKIP_FFMPEG_INSTALL = "1"   only check for ffmpeg, never install a package
param(
    [string]$ServerHost = "host.example.ts.net"
)
$ErrorActionPreference = "Stop"

$RerunVersion = "0.37.1"
$GrpcPort = 9876

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
        [Environment]::GetEnvironmentVariable("Path", "User")
}

# 1. uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "== Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Refresh-Path
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv still not on PATH; open a new terminal and re-run this script"
    }
}

# 2. rerun viewer, pinned
Write-Host "== Installing rerun-sdk==$RerunVersion (must match the server)"
uv tool install --force "rerun-sdk==$RerunVersion"
$toolBin = (uv tool dir --bin).Trim()
if (($env:Path -split ";") -notcontains $toolBin) {
    $env:Path = "$toolBin;$env:Path"
    Write-Host "note: $toolBin is not on your PATH; run 'uv tool update-shell' once, then open a new terminal"
}

# 3. ffmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if ($env:SKIP_FFMPEG_INSTALL -eq "1") {
        Write-Host "== ffmpeg not found; SKIP_FFMPEG_INSTALL=1 so not installing (video panes will not decode)"
    } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "== Installing ffmpeg (winget Gyan.FFmpeg)"
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        Refresh-Path
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "== Installing ffmpeg (choco)"
        choco install -y ffmpeg
        Refresh-Path
    } else {
        Write-Warning "Neither winget nor choco found. Download ffmpeg >= 5.1 from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH."
    }
}

# 4. verify
Write-Host "== Versions"
$rerunVersionOutput = & rerun --version
$rerunVersionOutput | ForEach-Object { Write-Host $_ }
$reported = ($rerunVersionOutput | Select-Object -First 1) -split "\s+" | Select-Object -Index 1
if ($reported -ne $RerunVersion) {
    throw "rerun reports $reported, expected $RerunVersion (another rerun earlier on PATH?)"
}
if (-not ($rerunVersionOutput -match "Video features:.*ffmpeg")) {
    Write-Warning "this rerun build lists no ffmpeg video feature; H.264 will not decode"
}
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    & ffmpeg -version | Select-Object -First 1 | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "ffmpeg: not on PATH (Rerun will show 'failed to create video decoder' on H.264 panes)"
}

Write-Host ""
Write-Host "== Ready. Connect to the server (it must be running scripts/serve_review_over_tailscale.sh):"
Write-Host "  rerun --connect rerun+http://${ServerHost}:${GrpcPort}/proxy"
Write-Host "  (or: $PSScriptRoot\connect.ps1 $ServerHost)"
Write-Host "If the server instead serves the .rrd file over plain http, open it directly:"
Write-Host "  rerun http://${ServerHost}:<port>/<recording>.rrd"
