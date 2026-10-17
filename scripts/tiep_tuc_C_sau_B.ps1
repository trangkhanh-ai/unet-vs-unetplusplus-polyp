param([int]$PollSeconds = 20)

# 1. Xác định các đường dẫn từ vị trí script, không phụ thuộc thư mục đang mở.
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$PythonPath = Join-Path (Split-Path -Parent $RepoRoot) '.venv-polyp/Scripts/python.exe'
$BSummaryPath = Join-Path $RepoRoot 'runs/B_unetplusplus_seed42/summary.json'
$CRunPath = Join-Path $RepoRoot 'runs/C_unetplusplus_ds_seed42'
$FrozenSourcePath = Join-Path $RepoRoot 'results/verification_bf16_v2/protocol_source_hashes.json'
$StdoutPath = Join-Path $RepoRoot 'runs/C_unetplusplus_ds_seed42_stdout.log'
$StderrPath = Join-Path $RepoRoot 'runs/C_unetplusplus_ds_seed42_stderr.log'
$LaunchInfoPath = Join-Path $RepoRoot 'runs/C_unetplusplus_ds_seed42_launch.json'
$ConfigHashes = @{
    'configs/unet.json' = '95d33123fe92406a00084417077e216648b8df89788e816de5e1e97408270903'
    'configs/unetplusplus.json' = '97363c58288257b96648cdf050c51f0b7d088d019e43034dd79a8a5338f6b31c'
    'configs/unetplusplus_ds.json' = 'ac5cdba3d1bf71b077569ebbfeb80e8e4176c0c90001767d29225a2ead04d537'
}

function Write-ProgressLog([string]$Message) {
    Write-Output ((Get-Date).ToUniversalTime().ToString('o') + ' | ' + $Message)
}

function Assert-FrozenProtocol {
    $Frozen = Get-Content -LiteralPath $FrozenSourcePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (@($Frozen.PSObject.Properties).Count -ne 8) {
        throw 'The frozen protocol must list exactly eight execution source files.'
    }
    foreach ($Entry in $Frozen.PSObject.Properties) {
        $Actual = (Get-FileHash -LiteralPath (Join-Path $RepoRoot $Entry.Name) -Algorithm SHA256).Hash
        if ($Actual.ToLowerInvariant() -ne $Entry.Value) {
            throw ('Execution source changed: ' + $Entry.Name)
        }
    }
    foreach ($Name in $ConfigHashes.Keys) {
        $Actual = (Get-FileHash -LiteralPath (Join-Path $RepoRoot $Name) -Algorithm SHA256).Hash
        if ($Actual.ToLowerInvariant() -ne $ConfigHashes[$Name]) {
            throw ('Official config changed: ' + $Name)
        }
        $Config = Get-Content -LiteralPath (Join-Path $RepoRoot $Name) -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $Config.amp -or $Config.amp_dtype -ne 'bfloat16') {
            throw ('Official config must use BF16 AMP: ' + $Name)
        }
    }
}

# 2. Chỉ cho một watcher sở hữu quyền tự khởi chạy C.
$Mutex = [System.Threading.Mutex]::new($false, 'Local\PolypOfficialBF16LaunchC')
$HasMutex = $false
try {
    $HasMutex = $Mutex.WaitOne(0)
    if (-not $HasMutex) {
        throw 'Another watcher already owns the launch of C.'
    }
    Assert-FrozenProtocol
    Write-ProgressLog 'Watcher ready; waiting for B to complete or early stop.'

    # 3. Đợi trạng thái đã lưu của B; không đọc ảnh test hay thay đổi B.
    $LastEpoch = -1
    while ($true) {
        if (Test-Path -LiteralPath $CRunPath) {
            Write-ProgressLog 'C output already exists; refusing to start a duplicate.'
            return
        }
        if (Test-Path -LiteralPath $BSummaryPath) {
            $B = Get-Content -LiteralPath $BSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($B.epochs_completed -ne $LastEpoch) {
                $LastEpoch = $B.epochs_completed
                Write-ProgressLog ('B status=' + $B.status + '; epoch=' + $LastEpoch)
            }
            if ($B.status -in @('completed', 'early_stopped')) { break }
            if ($B.status -ne 'training') { throw ('Unexpected B status: ' + $B.status) }
        }
        Start-Sleep -Seconds $PollSeconds
    }

    # 4. Kiểm tra lại source/config trước khi bắt đầu một lượt C hoàn toàn mới.
    Assert-FrozenProtocol
    if ($B.training_precision -ne 'amp_bfloat16' -or $B.test_images_read -ne 0) {
        throw 'B must have BF16 training and must not have read test images.'
    }
    $Existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*project.py train*C_unetplusplus_ds_seed42*' }
    if ($Existing) { throw 'A C training process already exists.' }
    if (Test-Path -LiteralPath $CRunPath) { throw 'C output appeared before launch.' }

    # 5. Chạy C nền với log riêng. Không tạo trước C folder vì trainer yêu cầu
    # output chưa tồn tại khi fresh run; log nằm cạnh folder đó trong runs/.
    $Arguments = @('-u', '-X', 'utf8', '-B', 'project.py', 'train', '--config',
        'configs/unetplusplus_ds.json', '--output', 'runs/C_unetplusplus_ds_seed42',
        '--device', 'cuda')
    $Process = Start-Process -FilePath $PythonPath -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    $Info = [ordered]@{
        started_utc = (Get-Date).ToUniversalTime().ToString('o')
        launcher_pid = $Process.Id
        python = $PythonPath
        arguments = $Arguments
        stdout = $StdoutPath
        stderr = $StderrPath
        protocol = 'v2 BF16 fresh; no resume from FP16 history'
    }
    $InfoJson = $Info | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($LaunchInfoPath, $InfoJson, [System.Text.UTF8Encoding]::new($false))
    Write-ProgressLog ('C started; launcher PID=' + $Process.Id)
} finally {
    if ($HasMutex) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
