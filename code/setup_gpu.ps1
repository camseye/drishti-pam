<#
.SYNOPSIS
  Install the right CUDA-enabled torch wheel for Drishti-PAM on Windows.

.DESCRIPTION
  Detects whether a CUDA-capable GPU is present, queries `nvidia-smi` for the
  driver-supported CUDA runtime version, and installs the matching torch wheel
  from PyTorch's CUDA index. Then installs every other project dependency from
  requirements.txt. Falls back to the CPU-only build if no GPU is found and
  warns that fine-tuning will be slow.

  Tested with Python 3.13 and 3.14, torch 2.7-2.9, transformers v5.5+ on
  Windows 10 / 11 (May 2026).

.PARAMETER CudaVersion
  Force a specific CUDA wheel index ('cu126', 'cu128', 'cu130'). If not
  specified, the script auto-picks based on the installed driver.

.PARAMETER Force
  Reinstall torch even if a CUDA build is already present.

.PARAMETER HfToken
  Hugging Face access token. Saves it to .hf_token in the project root so
  every later run uses authenticated downloads (faster, no anonymous rate
  limits). Leave empty to skip; the script will prompt interactively if
  no token is currently configured.

.PARAMETER NoTokenPrompt
  Skip the interactive HF token prompt entirely.

.EXAMPLE
  # Auto-detect, install the right wheel, prompt for token, then install the rest:
  .\setup_gpu.ps1

.EXAMPLE
  # Force CUDA 12.8 and pre-supply token:
  .\setup_gpu.ps1 -CudaVersion cu128 -HfToken hf_xxxxxxxxxxxxxx
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cu126', 'cu128', 'cu130', 'cpu')]
    [string]$CudaVersion = 'auto',

    [switch]$Force,

    [string]$HfToken = '',

    [switch]$NoTokenPrompt
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

# --- 1. Detect Python --------------------------------------------------- #
Write-Section "1. Python interpreter"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Write-Error "python.exe not on PATH. Install Python 3.13 or 3.14 from python.org first."
    exit 1
}
$pyVersion = & python --version
Write-Host "  $pyVersion ($py)"
$pyMajorMinor = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ($pyMajorMinor -notmatch '^3\.(13|14)$') {
    Write-Warning "  This project is tested on Python 3.13 and 3.14. You have $pyMajorMinor — proceeding anyway."
}

# --- 2. Detect GPU + driver-supported CUDA ------------------------------ #
Write-Section "2. GPU detection"
$nvidiaSmi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
$detectedCuda = $null
if ($nvidiaSmi) {
    try {
        $smi = & nvidia-smi 2>$null
        $gpuLine = ($smi | Select-String -Pattern '\| +(\d+) +.* (\d+\.\d+) +\|' | Select-Object -First 1).Line
        $cudaLine = $smi | Select-String -Pattern 'CUDA Version: ?(\d+\.\d+)' | Select-Object -First 1
        if ($cudaLine) {
            $detectedCuda = $cudaLine.Matches[0].Groups[1].Value
        }
        $gpuName = ($smi | Select-String -Pattern '\| +\d+ +([^|]+?) +(?:On|Off) +\|' | Select-Object -First 1)
        if ($gpuName) {
            Write-Host ("  Detected: " + $gpuName.Matches[0].Groups[1].Value.Trim())
        } else {
            $smi | Select-String -Pattern 'NVIDIA' | Select-Object -First 1 | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
        }
        if ($detectedCuda) {
            Write-Host "  Driver supports CUDA up to: $detectedCuda"
        }
    } catch {
        Write-Warning "  nvidia-smi found but failed to parse output: $_"
    }
} else {
    Write-Host "  nvidia-smi not on PATH — assuming no GPU (or driver not installed)."
}

# --- 3. Pick the right wheel index ------------------------------------- #
Write-Section "3. Pick wheel index"
$wheelIndex = $null
if ($CudaVersion -eq 'auto') {
    if (-not $detectedCuda) {
        Write-Warning "  No GPU detected. Falling back to CPU-only torch."
        Write-Warning "  Local DistilBERT fine-tune will take ~25-40 minutes."
        Write-Warning "  Consider running on Kaggle (code/notebooks/cloud_run.ipynb) instead."
        $wheelIndex = 'cpu'
    } else {
        # Driver supports CUDA up to $detectedCuda; pick the highest stable wheel <= driver max.
        # PyTorch 2.9 stable wheels (May 2026): cu126, cu128. cu130 is preview.
        $cudaMajor, $cudaMinor = $detectedCuda -split '\.' | ForEach-Object { [int]$_ }
        if ($cudaMajor -ge 13) {
            $wheelIndex = 'cu130'
            Write-Host "  Driver supports CUDA 13.x — using preview wheel 'cu130'."
        } elseif ($cudaMajor -eq 12 -and $cudaMinor -ge 8) {
            $wheelIndex = 'cu128'
            Write-Host "  Driver supports CUDA 12.8+ — using stable wheel 'cu128'."
        } elseif ($cudaMajor -eq 12 -and $cudaMinor -ge 6) {
            $wheelIndex = 'cu126'
            Write-Host "  Driver supports CUDA 12.6+ — using stable wheel 'cu126'."
        } else {
            Write-Warning "  Driver supports only CUDA $detectedCuda. Update your NVIDIA driver to >= 555.85 for CUDA 12.6 wheels."
            Write-Warning "  Falling back to CPU torch."
            $wheelIndex = 'cpu'
        }
    }
} else {
    $wheelIndex = $CudaVersion
    Write-Host "  Using user-specified wheel index: $wheelIndex"
}

# --- 4. Check existing torch ------------------------------------------- #
Write-Section "4. Existing torch install"
$torchInfo = $null
try {
    $torchInfo = & python -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>$null
} catch { }
if ($torchInfo) {
    $torchVersion, $torchCuda = $torchInfo -split '\|'
    Write-Host "  torch $torchVersion (CUDA available: $torchCuda)"
    $needsReinstall = $Force -or ($wheelIndex -ne 'cpu' -and $torchCuda -ne 'True') -or ($torchVersion -like '*+cpu*' -and $wheelIndex -ne 'cpu')
} else {
    Write-Host "  torch not installed."
    $needsReinstall = $true
}

# --- 5. Install torch -------------------------------------------------- #
if ($needsReinstall) {
    Write-Section "5. Install torch ($wheelIndex)"
    if ($torchInfo) {
        Write-Host "  Uninstalling old torch..."
        & python -m pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null
    }
    if ($wheelIndex -eq 'cpu') {
        & python -m pip install --upgrade "torch>=2.7"
    } else {
        $url = "https://download.pytorch.org/whl/$wheelIndex"
        Write-Host "  pip install torch --index-url $url"
        & python -m pip install --upgrade "torch>=2.7" --index-url $url
    }
} else {
    Write-Section "5. Install torch (skipped — already correct, use -Force to override)"
}

# --- 6. Install everything else --------------------------------------- #
Write-Section "6. Install other dependencies"
$reqs = Join-Path $projectRoot 'requirements.txt'
if (-not (Test-Path $reqs)) {
    Write-Error "requirements.txt not found at $reqs"
    exit 1
}
# requirements.txt also lists torch, but pip will skip it because it's already
# satisfied by what we just installed.
& python -m pip install --upgrade -r $reqs

# --- 7. Hugging Face token -------------------------------------------- #
Write-Section "7. Hugging Face access token"
$tokenFile = Join-Path $projectRoot '.hf_token'
$existingToken = $env:HF_TOKEN
if (-not $existingToken -and (Test-Path $tokenFile)) {
    $existingToken = (Get-Content $tokenFile -Raw).Trim()
}

if ($HfToken) {
    # Caller supplied a token on the command line — save it.
    Set-Content -Path $tokenFile -Value $HfToken.Trim() -Encoding ASCII -NoNewline
    Write-Host "  Saved token to $tokenFile"
    $env:HF_TOKEN = $HfToken.Trim()
} elseif ($existingToken) {
    $masked = if ($existingToken.Length -gt 8) {
        $existingToken.Substring(0, 4) + '...' + $existingToken.Substring($existingToken.Length - 4, 4)
    } else { '(short)' }
    Write-Host "  Existing token found ($masked) — keeping it."
} elseif (-not $NoTokenPrompt) {
    Write-Host ""
    Write-Host "  A Hugging Face token avoids anonymous-download rate limits"
    Write-Host "  and can be 5-10x faster on the first model pull."
    Write-Host "  Get one at https://huggingface.co/settings/tokens (read access is enough)."
    $entered = Read-Host "  Paste token (or press Enter to skip)"
    if ($entered) {
        Set-Content -Path $tokenFile -Value $entered.Trim() -Encoding ASCII -NoNewline
        Write-Host "  Saved token to $tokenFile" -ForegroundColor Green
        $env:HF_TOKEN = $entered.Trim()
    } else {
        Write-Host "  Skipped — downloads will use anonymous endpoints."
    }
} else {
    Write-Host "  Skipped (-NoTokenPrompt)."
}

# --- 8. Verify -------------------------------------------------------- #
Write-Section "8. Verify"
$check = & python -c "import torch, transformers, datasets, accelerate; print(f'  torch={torch.__version__}'); print(f'  transformers={transformers.__version__}'); print(f'  datasets={datasets.__version__}'); print(f'  accelerate={accelerate.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"
$check
Write-Host ""
Write-Host "Setup complete. Run the pipeline interactively with:" -ForegroundColor Green
Write-Host "  python smoke.py" -ForegroundColor Green
Write-Host "Or non-interactively with:" -ForegroundColor Green
Write-Host "  python smoke.py --backend huggingface --expand-factor 2" -ForegroundColor Green
