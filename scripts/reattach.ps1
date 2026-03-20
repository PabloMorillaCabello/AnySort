# =============================================================================
# Reattach Orbbec Gemini camera to WSL2 via usbipd-win
#
# Prerequisites:
#   - usbipd-win installed: winget install usbipd
#   - WSL2 running
#
# Usage (run in PowerShell as Administrator):
#   .\scripts\reattach.ps1
#
# You can also schedule this via Task Scheduler to run at login.
# =============================================================================

$VID = "2bc5"    # Orbbec vendor ID
$PID = "0670"    # Gemini 2 product ID (change if using a different model)

Write-Host "Looking for Orbbec device ($VID`:$PID)..." -ForegroundColor Cyan

# Find the device in usbipd list
$device = usbipd list | Select-String "$VID`:$PID"

if (-not $device) {
    Write-Host "ERROR: Orbbec device not found. Is the camera plugged in?" -ForegroundColor Red
    exit 1
}

# Extract BUSID (first column, format like "1-3")
$busid = ($device -split '\s+')[0]
Write-Host "Found device at BUSID: $busid" -ForegroundColor Green

# Bind the device (makes it available for sharing — idempotent)
Write-Host "Binding device..." -ForegroundColor Yellow
usbipd bind --busid $busid 2>$null

# Attach to WSL2
Write-Host "Attaching to WSL2..." -ForegroundColor Yellow
usbipd attach --wsl --busid $busid

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Orbbec camera attached to WSL2" -ForegroundColor Green
    Write-Host "Verify in WSL2 with: lsusb | grep Orbbec" -ForegroundColor Cyan
} else {
    Write-Host "FAILED: Could not attach device. Is WSL2 running?" -ForegroundColor Red
    exit 1
}
