param([ValidateRange(1024, 65535)][int]$Port = 8501)
$ErrorActionPreference = 'Stop'

# 1. Tìm repo và môi trường Python trên máy hiện tại.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$PolypPython = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PolypPython)) {
    $PolypPython = Join-Path $WorkspaceRoot '.venv-polyp/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PolypPython)) {
    throw 'Chưa có Python của đồ án. Đọc mục cài môi trường trong README.md.'
}

# 2. Mở ứng dụng cục bộ. Giữ cửa sổ này; Ctrl+C để dừng.
Write-Host "Mở trình duyệt tại http://127.0.0.1:$Port"
Write-Host 'Chọn model, tải ảnh nội soi và xem mask/overlay. Ctrl+C để dừng.'
Push-Location -LiteralPath $RepoRoot
try {
    & $PolypPython -X utf8 -B project.py demo --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Demo chưa chạy được. Đọc lỗi ngay phía trên.' }
}
finally { Pop-Location }
