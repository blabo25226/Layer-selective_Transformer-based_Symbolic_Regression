# Create hardlinks so TPSR can reuse existing checkpoints without duplicate downloads.
# Run from repository root in PowerShell.

$root = Split-Path -Parent $PSScriptRoot

function Ensure-HardLink($link, $target) {
    if (-not (Test-Path $target)) {
        Write-Host "SKIP missing target: $target"
        return
    }
    $dir = Split-Path -Parent $link
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    if (Test-Path $link) {
        Write-Host "OK exists: $link"
        return
    }
    cmd /c mklink /H "$link" "$target" | Out-Null
    Write-Host "LINKED $link -> $target"
}

Ensure-HardLink `
    "$root\TPSR\nesymres\weights\10M.ckpt" `
    "$root\NSRS\weights\10M.ckpt"

Ensure-HardLink `
    "$root\TPSR\symbolicregression\weights\model.pt" `
    "$root\TPSR\symbolicregression\weights\model1.pt"
