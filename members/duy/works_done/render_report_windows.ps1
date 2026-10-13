# Render báo cáo bằng Word native trên Windows; chỉ mở tài liệu đầu vào của repo.
param(
    [Parameter(Mandatory=$true)][string]$Document,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$SourcePath = (Resolve-Path -LiteralPath $Document).Path
$OutputPath = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$PdfPath = Join-Path $OutputPath 'report.pdf'
$WordApp = $null
$ReportDoc = $null
try {
    $WordApp = New-Object -ComObject Word.Application
    if ($WordApp.Documents.Count -ne 0) {
        throw 'Không nhận một phiên Word mới; dừng để bảo toàn tài liệu đang mở.'
    }
    $WordApp.Visible = $false
    $WordApp.DisplayAlerts = 0
    $WordApp.AutomationSecurity = 3
    $ReportDoc = $WordApp.Documents.Open($SourcePath, $false, $true, $false)
    $ReportDoc.Repaginate()
    $PageCount = $ReportDoc.ComputeStatistics(2)
    $ReportDoc.ExportAsFixedFormat($PdfPath, 17)
    [pscustomobject]@{renderer='Microsoft Word native';version=$WordApp.Version;pages=$PageCount;pdf=$PdfPath} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputPath 'render_metadata.json') -Encoding UTF8
    Write-Output "Rendered pages: $PageCount"
}
finally {
    if ($null -ne $ReportDoc) {
        $ReportDoc.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($ReportDoc)
    }
    if ($null -ne $WordApp) {
        if ($WordApp.Documents.Count -eq 0) { $WordApp.Quit(0) }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($WordApp)
    }
}
$BundledPoppler = 'C:/Users/dzyuu/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
if (-not (Test-Path -LiteralPath $BundledPoppler)) { throw 'Thiếu Poppler bundled để rasterize PDF' }
& $BundledPoppler -r 120 -png $PdfPath (Join-Path $OutputPath 'page')
if ($LASTEXITCODE -ne 0) { throw 'Không xuất được PNG để kiểm tra bố cục' }
