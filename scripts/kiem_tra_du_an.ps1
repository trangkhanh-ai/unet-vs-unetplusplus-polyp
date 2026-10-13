$ErrorActionPreference = 'Stop'

# 1. Tìm repo và môi trường Python trên máy hiện tại.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$PolypPython = Join-Path $WorkspaceRoot '.venv-polyp/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PolypPython)) {
    $PolypPython = Join-Path $RepoRoot '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PolypPython)) {
    throw 'Chưa có Python của đồ án. Đọc docs/HOC_VA_CHAY_CODE_CHO_DUY.md, mục 2.'
}

# 2. Mỗi lần kiểm tra dùng folder mới, giữ minh chứng chính thức.
$OutputFolder = 'results/verification_local_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
Push-Location -LiteralPath $RepoRoot
try {
    if (Test-Path -LiteralPath $OutputFolder) { throw "Folder đã tồn tại: $OutputFolder" }
    & $PolypPython -X utf8 -B project.py test --output $OutputFolder
    if ($LASTEXITCODE -ne 0) { throw 'Có kiểm tra thất bại. Đọc lỗi ngay phía trên.' }
    Write-Host "Xem $OutputFolder/test_summary.json"
    Write-Host 'Đây là kiểm tra code; điểm model nằm trong results/comparison.'
}
finally { Pop-Location }
