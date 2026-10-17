# Xuất slide thành PNG bằng một phiên PowerPoint riêng để kiểm tra bố cục.
param(
    [Parameter(Mandatory=$true)][string]$Presentation,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$SourcePath = (Resolve-Path -LiteralPath $Presentation).Path
$OutputPath = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$PowerPointApp = $null
$SlideDeck = $null
try {
    $PowerPointApp = New-Object -ComObject PowerPoint.Application
    if ($PowerPointApp.Presentations.Count -ne 0) {
        throw 'Không nhận một phiên PowerPoint mới; dừng để bảo toàn tài liệu đang mở.'
    }
    # ReadOnly true, Untitled false, WithWindow false.
    $SlideDeck = $PowerPointApp.Presentations.Open($SourcePath, -1, 0, 0)
    $SlideCount = $SlideDeck.Slides.Count
    foreach ($Slide in $SlideDeck.Slides) {
        $ImagePath = Join-Path $OutputPath ('slide-{0:D2}.png' -f $Slide.SlideIndex)
        $Slide.Export($ImagePath, 'PNG', 1600, 900)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($Slide)
    }
    [pscustomobject]@{renderer='Microsoft PowerPoint native';version=$PowerPointApp.Version;slides=$SlideCount} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputPath 'render_metadata.json') -Encoding UTF8
    Write-Output "Rendered slides: $SlideCount"
}
finally {
    if ($null -ne $SlideDeck) {
        $SlideDeck.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($SlideDeck)
    }
    if ($null -ne $PowerPointApp) {
        if ($PowerPointApp.Presentations.Count -eq 0) { $PowerPointApp.Quit() }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($PowerPointApp)
    }
}
