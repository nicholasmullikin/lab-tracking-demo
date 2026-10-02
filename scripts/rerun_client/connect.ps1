# Open the native Rerun viewer against the review server's gRPC proxy.
# Usage: powershell -ExecutionPolicy Bypass -File connect.ps1 -ServerHost host.example.ts.net
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ServerHost
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command rerun -ErrorAction SilentlyContinue)) {
    throw "rerun not on PATH; run $PSScriptRoot\install.ps1 first"
}
& rerun --connect "rerun+http://${ServerHost}:9876/proxy"
